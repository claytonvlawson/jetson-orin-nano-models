#!/usr/bin/env python3
"""
context_guard.py - transparent context-window compactor for llama.cpp router.

Sits in front of llama-server (default 127.0.0.1:8080) and watches the token
count of every /v1/chat/completions request. When the next turn is projected to
overrun the model's context window, it:

  1. folds the older turns into a markdown summary,
  2. writes the full folded transcript to ~/contexts/compact-<ts>.md,
  3. rewrites the request as system prompt + compacted-context block + the most
     recent turns, i.e. continues in a fresh window,
  4. forwards the rewritten request upstream and streams the reply back.

Token counts come from the model's own tokenizer (POST /tokenize), so there is
no char/token guessing.

Endpoints: everything is proxied. Two local extras:
  GET  /guard/status    -> config, last compaction, live token estimate
  POST /guard/compact   -> force a compaction dry-run on the posted body
Header X-Force-Compact: 1 forces compaction on a single request.
"""

import datetime
import json
import os
import re
import sys
import threading
import time
import traceback
import http.client
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

CONFIG_PATH = os.environ.get(
    "GUARD_CONFIG", os.path.expanduser("~/jetson-context-guard/config.json")
)

DEFAULTS = {
    "listen_host": "127.0.0.1",
    "listen_port": 8090,
    "upstream_host": "127.0.0.1",
    "upstream_port": 8080,
    "model": "",
    "n_ctx": 32768,
    "n_ctx_by_model": {},
    "trigger_ratio": 0.80,
    # Absolute token trigger. When > 0 it wins over trigger_ratio, so a model
    # with a 32k window can be compacted at e.g. 16000 tokens instead of 80%.
    "trigger_tokens": 0,
    "reserve_tokens": 768,
    "keep_recent_messages": 6,
    # Reasoning models spend their whole budget thinking when handed a resumed
    # window, which returns an empty answer with finish_reason=length. "auto"
    # suppresses thinking only when the client asked for fewer tokens than
    # thinking_budget_threshold (too little to finish a trace anyway).
    "compacted_thinking": "auto",
    "thinking_budget_threshold": 1024,
    "upstream_retries": 3,
    "summary_max_tokens": 1200,
    "summary_temperature": 0.2,
    "output_dir": os.path.expanduser("~/contexts"),
    "max_compactions_per_request": 3,
    "upstream_timeout": 900,
}

HOP_BY_HOP = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailers",
    "transfer-encoding",
    "upgrade",
}

COMPACT_MARKER = "## Compacted context (continuation)"

SUMMARY_SYSTEM = (
    "You compress conversation history so an assistant can continue the work in a "
    "fresh context window. Write dense markdown. Preserve: the user's goal, "
    "decisions made and why, every concrete identifier verbatim (file paths, "
    "commands, model names, URLs, numbers, ports, token counts, error strings), "
    "what was already tried and failed, and what is still open. Drop pleasantries "
    "and repetition. Use only facts written in the transcript: if something is "
    "not stated, leave it out rather than filling the gap from general knowledge "
    "(no added hardware specs, versions, or explanations). Copy identifiers "
    "character for character. "
    "Use these sections when they apply: Goal, Decisions, Artifacts, Tried and "
    "Failed, Open Questions, Next Step. Target at most 400 words."
)

_LOCK = threading.Lock()
_STATE = {
    "started": None,
    "requests": 0,
    "compactions": 0,
    "last_compaction": None,
    "last_file": None,
    "last_tokens_before": None,
    "last_tokens_after": None,
}


def load_config():
    cfg = dict(DEFAULTS)
    try:
        with open(CONFIG_PATH) as fh:
            cfg.update(json.load(fh))
    except FileNotFoundError:
        log("config not found at %s, using defaults" % CONFIG_PATH)
    except Exception as exc:  # malformed json must not take the proxy down
        log("config error (%s), using defaults" % exc)
    return cfg


CFG = load_config()


def trigger_tokens_for(n_ctx):
    """Absolute trigger_tokens wins over trigger_ratio when set."""
    abs_t = int(CFG.get("trigger_tokens") or 0)
    if abs_t > 0:
        return abs_t
    return int(n_ctx * float(CFG["trigger_ratio"]))


def log(msg):
    line = "%s %s" % (datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), msg)
    print(line, flush=True)
    try:  # the user journal is empty on this Jetson, so keep our own file
        path = os.path.join(CFG["output_dir"], "guard.log")
        if os.path.exists(path) and os.path.getsize(path) > 5_000_000:
            os.replace(path, path + ".1")
        with open(path, "a") as fh:
            fh.write(line + "\n")
    except Exception:
        pass


def log_event(event):
    """Append one JSON line to <output_dir>/guard.jsonl for later inspection."""
    try:
        os.makedirs(CFG["output_dir"], exist_ok=True)
        event["ts"] = datetime.datetime.now().astimezone().isoformat(timespec="seconds")
        with open(os.path.join(CFG["output_dir"], "guard.jsonl"), "a") as fh:
            fh.write(json.dumps(event) + "\n")
    except Exception as exc:
        log("log_event failed: %s" % exc)


def upstream(method, path, body=None, headers=None, timeout=None):
    """One-shot call to llama-server. Returns (status, headers dict, bytes)."""
    conn = http.client.HTTPConnection(
        CFG["upstream_host"],
        CFG["upstream_port"],
        timeout=timeout or CFG["upstream_timeout"],
    )
    send = {"Content-Type": "application/json"}
    if headers:
        send.update(headers)
    try:
        conn.request(method, path, body=body, headers=send)
        resp = conn.getresponse()
        data = resp.read()
        return resp.status, dict(resp.getheaders()), data
    finally:
        conn.close()


def count_tokens(text, model):
    """Exact token count from the model's tokenizer, with a coarse fallback."""
    if not text:
        return 0
    try:
        payload = json.dumps({"model": model, "content": text}).encode()
        status, _, data = upstream("POST", "/tokenize", payload, timeout=120)
        if status == 200:
            return len(json.loads(data).get("tokens", []))
    except Exception as exc:
        log("tokenize failed (%s), estimating" % exc)
    return max(1, len(text) // 4)


def messages_tokens(messages, model):
    """Approximate chat-template overhead per message is ~4 tokens."""
    joined = "\n".join(m.get("content") or "" for m in messages if isinstance(m, dict))
    return count_tokens(joined, model) + 4 * len(messages)


def is_system(msg):
    return isinstance(msg, dict) and msg.get("role") == "system"


def split_messages(messages):
    """(system messages, foldable middle, recent tail)."""
    systems = []
    rest = list(messages)
    while rest and is_system(rest[0]):
        systems.append(rest.pop(0))

    keep = int(CFG["keep_recent_messages"])
    if keep < 0:
        keep = 0
    if len(rest) <= keep:
        return systems, [], rest

    head, tail = rest[: len(rest) - keep], rest[len(rest) - keep:]
    # a tool result may not lead a window without its call
    while tail and isinstance(tail[0], dict) and tail[0].get("role") == "tool":
        head.append(tail.pop(0))
    return systems, head, tail


def render_transcript(messages):
    out = []
    for i, msg in enumerate(messages, 1):
        role = msg.get("role", "unknown")
        content = msg.get("content") or ""
        if isinstance(content, list):  # multimodal / content parts
            content = "\n".join(
                part.get("text", "") for part in content if isinstance(part, dict)
            )
        extra = ""
        if msg.get("tool_calls"):
            extra = "\n[tool_calls] " + json.dumps(msg["tool_calls"])[:2000]
        if msg.get("name"):
            role = "%s (%s)" % (role, msg["name"])
        out.append("### %d. %s\n\n%s%s\n" % (i, role, content.strip(), extra))
    return "\n".join(out)


def _summarize_once(system_prompt, prompt, model, max_tokens):
    payload = json.dumps(
        {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": int(max_tokens),
            "temperature": float(CFG["summary_temperature"]),
            "stream": False,
            # Qwen3-style templates: keep the budget for the summary, not for CoT
            "chat_template_kwargs": {"enable_thinking": False},
        }
    ).encode()
    status, _, data = upstream("POST", "/v1/chat/completions", payload, timeout=1800)
    if status != 200:
        raise RuntimeError(
            "summarizer returned %s: %s" % (status, data[:300].decode(errors="replace"))
        )
    body = json.loads(data)
    choice = body["choices"][0]
    text = (choice["message"].get("content") or "").strip()
    if not text:
        raise RuntimeError(
            "empty content (finish_reason=%s)" % choice.get("finish_reason")
        )
    return text


def summarize(chunk, model, part_label=""):
    """Summarize without letting a reasoning model spend the budget on CoT.

    A thinking model that answers with reasoning_content and empty content is
    useless here: the thinking trace is not a summary. Retry with thinking
    suppressed via /no_think and a larger budget, and fail loudly rather than
    silently persisting a chain-of-thought dump.
    """
    prompt = "TRANSCRIPT%s:\n\n%s" % (part_label, render_transcript(chunk))
    base = int(CFG["summary_max_tokens"])
    attempts = [
        (SUMMARY_SYSTEM, prompt, base),
        (SUMMARY_SYSTEM, "/no_think\n\n" + prompt, int(base * 1.5)),
        (SUMMARY_SYSTEM + "\n/no_think", prompt, int(base * 2)),
    ]
    last = None
    for system_prompt, text_prompt, budget in attempts:
        try:
            return _summarize_once(system_prompt, text_prompt, model, budget)
        except Exception as exc:
            last = exc
            log("summarizer attempt failed: %s" % exc)
    raise RuntimeError("summarizer produced no usable summary: %s" % last)


def group_by_budget(messages, model, budget):
    """Split foldable messages into chunks that each fit the summarizer window."""
    groups, current, used = [], [], 0
    for msg in messages:
        cost = messages_tokens([msg], model)
        if current and used + cost > budget:
            groups.append(current)
            current, used = [], 0
        current.append(msg)
        used += cost
    if current:
        groups.append(current)
    return groups


def write_markdown(meta, summary, folded, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    front = "\n".join("%s: %s" % (k, v) for k, v in meta.items())
    doc = [
        "---",
        front,
        "---",
        "",
        "# Compacted context",
        "",
        "## Summary",
        "",
        summary.strip(),
        "",
        "## Full transcript (folded portion)",
        "",
        render_transcript(folded).strip(),
        "",
    ]
    with open(path, "w") as fh:
        fh.write("\n".join(doc))


def summarize_grouped(messages, model, n_ctx):
    """Summarize a fold set, map-reducing when it exceeds the summarizer window."""
    budget = max(
        2048,
        int(n_ctx or CFG["n_ctx"])
        - int(CFG["summary_max_tokens"])
        - int(CFG["reserve_tokens"]),
    )
    groups = group_by_budget(messages, model, budget)
    if len(groups) == 1:
        return summarize(groups[0], model)
    parts = []
    for i, group in enumerate(groups, 1):
        parts.append(summarize(group, model, part_label=" PART %d/%d" % (i, len(groups))))
    merged = [{"role": "user", "content": "\n\n---\n\n".join(parts)}]
    return summarize(merged, model, part_label=" OF MERGED PARTIAL SUMMARIES")


def build_block(summary, path):
    return (
        "%s\n\nEarlier turns of this conversation were compacted to keep it inside the "
        "model's context window. The full transcript is saved at %s.\n\n%s\n\n"
        "Resume the conversation from this state. Answer the latest user message "
        "directly: do not restate this block, do not re-summarize it, and do not ask "
        "the user to repeat anything already captured above."
    ) % (COMPACT_MARKER, path, summary.strip())


def normalize_roles(msgs):
    """Collapse back-to-back same-role turns.

    The rewritten window is system + summary-user + ack-assistant + kept tail,
    and the tail often starts with an assistant turn, which yields two assistant
    messages in a row. Chat templates and reasoning models handle that badly, so
    merge them (system prompts stay separate and first).
    """
    out = []
    for msg in msgs:
        role = msg.get("role")
        if out and role != "system" and out[-1].get("role") == role:
            out[-1] = {"role": role,
                       "content": (out[-1].get("content") or "").rstrip()
                       + "\n\n" + (msg.get("content") or "").lstrip()}
        else:
            out.append({"role": role, "content": msg.get("content") or ""})
    return out


def compact(messages, model, tokens_before, n_ctx=None, trigger=None):
    """Fold older turns into a markdown checkpoint and rewrite the window.

    When `trigger` is given, keep folding further back until the rewritten
    conversation actually fits under it (bounded by max_compactions_per_request),
    so one oversized message cannot keep the window over budget. Later passes
    summarize only the newly folded messages and append, instead of re-reading
    the whole history.
    """
    systems, folded, tail = split_messages(messages)
    if not folded:
        raise RuntimeError("nothing foldable (conversation is already minimal)")

    folded, tail = list(folded), list(tail)
    max_passes = max(1, int(CFG["max_compactions_per_request"]))
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    path = os.path.join(CFG["output_dir"], "compact-%s.md" % stamp)

    pending, summary, passes, new_messages, after = list(folded), "", 0, None, None
    while True:
        passes += 1
        part = summarize_grouped(pending, model, n_ctx)
        summary = part if not summary else summary + "\n\n### Even earlier\n\n" + part
        ack = [{"role": "assistant",
                "content": "Understood. Continuing from the compacted context above."}]
        if tail and isinstance(tail[0], dict) and tail[0].get("role") == "assistant":
            ack = []  # the kept tail already opens with an assistant turn
        new_messages = normalize_roles(
            systems + [{"role": "user", "content": build_block(summary, path)}] + ack + tail
        )
        after = messages_tokens(new_messages, model)
        if trigger is None or after <= trigger or not tail or passes >= max_passes:
            break
        log("rewrite still %d tokens, over trigger %d -> folding deeper (pass %d)"
            % (after, trigger, passes + 1))
        pending = []
        while tail and len(pending) < 2:
            msg = tail.pop(0)
            pending.append(msg)
            folded.append(msg)
        while tail and isinstance(tail[0], dict) and tail[0].get("role") == "tool":
            folded.append(tail.pop(0))

    meta = {
        "model": model,
        "created": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "source": "context_guard.py",
        "n_ctx": int(n_ctx or CFG["n_ctx"]),
        "tokens_before": tokens_before,
        "tokens_after": after,
        "messages_before": len(messages),
        "messages_folded": len(folded),
        "messages_kept": len(tail),
        "summarizer_passes": passes,
    }
    write_markdown(meta, summary, folded, path)

    info = {
        "file": path,
        "folded": len(folded),
        "kept": len(tail),
        "summary_chars": len(summary),
        "passes": passes,
        "tokens_after": after,
    }
    return new_messages, info


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "context-guard/1.0"

    def log_message(self, fmt, *args):  # keep the journal quiet-ish
        log("%s %s" % (self.address_string(), fmt % args))

    # ---------- plumbing ----------

    def _read_body(self):
        length = int(self.headers.get("Content-Length") or 0)
        return self.rfile.read(length) if length else b""

    def _send_json(self, obj, status=200):
        data = json.dumps(obj, indent=2).encode() + b"\n"
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _forward(self, method, path, body):
        """Proxy to upstream, streaming the response when the client asked for it."""
        stream = False
        if body:
            try:
                stream = bool(json.loads(body).get("stream"))
            except Exception:
                stream = False

        conn = http.client.HTTPConnection(
            CFG["upstream_host"], CFG["upstream_port"], timeout=CFG["upstream_timeout"]
        )
        headers = {"Content-Type": self.headers.get("Content-Type", "application/json")}
        if self.headers.get("Authorization"):
            headers["Authorization"] = self.headers["Authorization"]
        if self.headers.get("Accept"):
            headers["Accept"] = self.headers["Accept"]
        # llama-server serves its web UI gzip-only and answers 415 to clients that
        # do not advertise gzip, so this header must survive the proxy or every
        # static asset (and the service worker) breaks behind the guard.
        if self.headers.get("Accept-Encoding"):
            headers["Accept-Encoding"] = self.headers["Accept-Encoding"]
        if self.headers.get("User-Agent"):
            headers["User-Agent"] = self.headers["User-Agent"]

        # The router restarts its model child on demand; a request that lands in
        # that window used to surface as an instant 502. Retry briefly instead.
        attempts = max(1, int(CFG.get("upstream_retries", 3)))
        resp = None
        last_exc = None
        for attempt in range(attempts):
            try:
                conn.request(method, path, body=body, headers=headers)
                resp = conn.getresponse()
                if resp.status in (502, 503) and attempt < attempts - 1:
                    last_exc = "HTTP %d" % resp.status
                    resp.read()
                    conn.close()
                    resp = None
                    log("upstream %s, retrying (%d/%d)" % (last_exc, attempt + 1, attempts))
                    time.sleep(min(6, 2 ** attempt))
                    conn = http.client.HTTPConnection(
                        CFG["upstream_host"], CFG["upstream_port"],
                        timeout=CFG["upstream_timeout"])
                    continue
                break
            except Exception as exc:
                last_exc = exc
                conn.close()
                if attempt < attempts - 1:
                    log("upstream error (%s), retrying (%d/%d)"
                        % (exc, attempt + 1, attempts))
                    time.sleep(min(6, 2 ** attempt))
                    conn = http.client.HTTPConnection(
                        CFG["upstream_host"], CFG["upstream_port"],
                        timeout=CFG["upstream_timeout"])
                    continue
                self._send_json({"error": {"message": "upstream unreachable: %s" % exc,
                                           "type": "context_guard_error"}}, 502)
                return
        if resp is None:
            self._send_json({"error": {"message": "upstream unreachable: %s" % last_exc,
                                       "type": "context_guard_error"}}, 502)
            return

        self.send_response(resp.status)
        for key, value in resp.getheaders():
            if key.lower() in HOP_BY_HOP or key.lower() in ("content-length", "date", "server"):
                continue
            self.send_header(key, value)
        for key, value in self._guard_headers():
            self.send_header(key, value)

        if stream:
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            try:
                while True:
                    chunk = resp.read(2048)
                    if not chunk:
                        break
                    self.wfile.write(b"%X\r\n%s\r\n" % (len(chunk), chunk))
                    self.wfile.flush()
                self.wfile.write(b"0\r\n\r\n")
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass
        else:
            data = resp.read()
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        conn.close()

    def _guard_headers(self):
        info = self._guard_info
        if not info:
            return [("X-Context-Guard", "passthrough")]
        out = [
            ("X-Context-Guard", "compacted"),
            ("X-Context-Guard-File", info["file"]),
            ("X-Context-Guard-Folded", str(info["folded"])),
            ("X-Context-Guard-Kept", str(info["kept"])),
        ]
        if info.get("tokens_before") is not None:
            out.append(("X-Context-Guard-Tokens-Before", str(info["tokens_before"])))
            out.append(("X-Context-Guard-Tokens-After", str(info["tokens_after"])))
        return out

    _guard_info = None

    # ---------- routing ----------

    def do_GET(self):
        if self.path.startswith("/guard/status"):
            self._send_json({
                "config": {k: CFG[k] for k in DEFAULTS},
                "state": _STATE,
                "trigger_tokens": trigger_tokens_for(CFG["n_ctx"]),
            })
            return
        self._guard_info = None
        self._forward("GET", self.path, None)

    def do_POST(self):
        body = self._read_body()

        if self.path.startswith("/guard/compact"):
            try:
                req = json.loads(body or b"{}")
                model = req.get("model") or CFG["model"]
                msgs = req.get("messages") or []
                before = messages_tokens(msgs, model)
                new_msgs, info = compact(msgs, model, before, req.get("n_ctx"))
                self._send_json({"ok": True, "info": info,
                                 "tokens_before": before,
                                 "tokens_after": messages_tokens(new_msgs, model),
                                 "messages": new_msgs})
            except Exception as exc:
                self._send_json({"ok": False, "error": str(exc)}, 500)
            return

        if not self.path.startswith("/v1/chat/completions"):
            self._guard_info = None
            self._forward("POST", self.path, body)
            return

        try:
            self._handle_chat(body)
        except Exception:
            log("chat handler crashed:\n%s" % traceback.format_exc())
            self._guard_info = None
            self._forward("POST", self.path, body)

    def _handle_chat(self, body):
        self._guard_info = None
        _STATE["requests"] += 1
        req = json.loads(body)
        model = req.get("model") or CFG["model"]
        messages = req.get("messages") or []
        if not messages:
            self._forward("POST", self.path, body)
            return

        n_ctx = int(
            self.headers.get("X-Guard-NCtx")
            or CFG.get("n_ctx_by_model", {}).get(model)
            or CFG["n_ctx"]
        )
        reserve = int(req.get("max_tokens") or CFG["reserve_tokens"])
        budget = n_ctx - reserve - 256
        trigger = min(budget, trigger_tokens_for(n_ctx))
        force = self.headers.get("X-Force-Compact") == "1"

        before = messages_tokens(messages, model)
        if not force and before <= trigger:
            log("pass  model=%s tokens=%d/%d (trigger %d)" % (model, before, n_ctx, trigger))
            self._forward("POST", self.path, body)
            return

        log("compact model=%s tokens=%d (trigger %d, forced=%s)"
            % (model, before, trigger, force))
        try:
            with _LOCK:
                new_messages, info = compact(messages, model, before, n_ctx, trigger)
        except Exception as exc:
            # A fold must never cost the client its turn: fall through to the
            # original request and let the model answer uncompacted.
            log("compaction skipped (%s) - forwarding original request" % exc)
            self._forward("POST", self.path, body)
            return
        req["messages"] = new_messages
        policy = str(CFG.get("compacted_thinking", "auto")).lower()
        want = int(req.get("max_tokens") or 0)
        threshold = int(CFG.get("thinking_budget_threshold", 1024))
        if policy == "off" or (policy == "auto" and want and want < threshold):
            kwargs = req.get("chat_template_kwargs") or {}
            if "enable_thinking" not in kwargs:
                kwargs["enable_thinking"] = False
                req["chat_template_kwargs"] = kwargs
                log("compacted turn: thinking suppressed (max_tokens=%d)" % want)
        after = messages_tokens(new_messages, model)
        info["tokens_before"] = before
        info["tokens_after"] = after
        self._guard_info = info

        _STATE["compactions"] += 1
        _STATE["last_compaction"] = datetime.datetime.now().astimezone().isoformat(
            timespec="seconds")
        _STATE["last_file"] = info["file"]
        _STATE["last_tokens_before"] = before
        _STATE["last_tokens_after"] = after
        log_event({"event": "compacted", "model": model, **info})
        log("compacted %d -> %d tokens, file %s" % (before, after, info["file"]))

        self._forward("POST", self.path, json.dumps(req).encode())


def main():
    _STATE["started"] = datetime.datetime.now().astimezone().isoformat(timespec="seconds")
    os.makedirs(CFG["output_dir"], exist_ok=True)
    srv = ThreadingHTTPServer((CFG["listen_host"], int(CFG["listen_port"])), Handler)
    log("context-guard listening on %s:%s -> %s:%s (n_ctx=%s, trigger %.0f%% = %d tokens)"
        % (CFG["listen_host"], CFG["listen_port"], CFG["upstream_host"],
           CFG["upstream_port"], CFG["n_ctx"], float(CFG["trigger_ratio"]) * 100,
           trigger_tokens_for(CFG["n_ctx"])))
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()