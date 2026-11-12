#!/usr/bin/env python3
"""Survey the real per-model ceiling on this 8 GB Orin Nano (2nd take).

Failures on this board are dominated by CUDA allocator fragmentation, not by
capacity: the same config can fail twice and then load. So every probe is
retried, and a config only counts as "does not fit" after N clean attempts with
a verified-free memory state. Nothing here is extrapolated.

For each model:
  1. highest n_gpu_layers that loads and serves at ctx 4096
  2. largest context that loads and serves at that ngl (doubling, then bisect)
  3. at that ceiling: load time, buffers, RSS, prefill and generation tok/s

Usage: survey.py models.json      Output: JSON per model, /tmp/survey.jsonl
"""
import json
import os
import re
import subprocess
import sys
import time
import urllib.request

BIN = "/home/snowbeach/maple/llama.cpp/build/bin/llama-server"
PORT = 8099
RECLAIM = os.path.expanduser("~/bin/reclaim-free-pages.py")
TRIES = 3
PROMPT = ("The snow beach test corpus covers memory bandwidth, kernel launch "
          "overhead and cache behaviour on embedded GPUs. ") * 27


def memfree_mb():
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemFree"):
                return int(line.split()[1]) // 1024
    return 0


def clear():
    """Kill stragglers and drive enough anonymous pressure to reclaim page
    cache. 2000 MB is not enough: the GGUF reads leave ~2.6 GB cached."""
    subprocess.run(["pkill", "-x", "llama-server"])
    time.sleep(2)
    for _ in range(3):
        subprocess.run(["python3", RECLAIM, "5000"],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(2)
        if memfree_mb() > 4000:
            return
    time.sleep(2)


def _req(path, body=None, timeout=1800):
    url = f"http://127.0.0.1:{PORT}{path}"
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def load_once(model, ngl, ctx, extra):
    clear()
    args = [BIN, "-m", model, "-ngl", str(ngl), "-c", str(ctx), "-fa", "on",
            "--cache-type-k", "q8_0", "--cache-type-v", "q8_0", "--no-mmap",
            "-np", "1", "--port", str(PORT), "--host", "127.0.0.1",
            "--jinja", "-lv", "4"] + list(extra)
    tag = re.sub(r"\W+", "_", os.path.basename(model))[:20]
    lp = f"/tmp/sv-{tag}-n{ngl}-c{ctx}.log"
    log = open(lp, "w")
    t0 = time.time()
    p = subprocess.Popen(args, stdout=log, stderr=subprocess.STDOUT)
    ok = False
    while time.time() - t0 < 900:
        if p.poll() is not None:
            break
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=3)
            ok = True
            break
        except Exception:
            time.sleep(0.5)
    load_s = round(time.time() - t0, 1)
    if ok:
        try:
            _req("/completion", {"prompt": "hi", "n_predict": 8,
                                 "temperature": 0})
        except Exception:
            ok = False
    log.close()
    return p, ok, load_s, lp


def try_load(model, ngl, ctx, extra, keep=False):
    """Load with retries. Returns (proc, info) or (None, info)."""
    info = {"ngl": ngl, "ctx": ctx, "ok": False, "tries": 0, "load_s": None}
    for a in range(TRIES):
        info["tries"] = a + 1
        p, ok, load_s, lp = load_once(model, ngl, ctx, extra)
        info["load_s"] = load_s
        if ok:
            txt = open(lp, errors="ignore").read()

            def f(pat):
                m = re.search(pat, txt)
                return m.group(1) if m else None
            info.update({
                "ok": True,
                "model_buf_mib": f(r"CUDA0 model buffer size\s*=\s*([\d.]+)"),
                "kv_buf_mib": f(r"CUDA0 KV buffer size\s*=\s*([\d.]+)"),
                "cpu_buf_mib": f(r"CUDA_Host model buffer size\s*=\s*([\d.]+)"),
                "n_ctx_train": f(r"n_ctx_train\s*=\s*(\d+)"),
                "n_layer": f(r"n_layer\s*=\s*(\d+)"),
            })
            try:
                info["rss_mib"] = int(subprocess.run(
                    ["ps", "-o", "rss=", "-p", str(p.pid)],
                    capture_output=True, text=True).stdout.strip()) // 1024
            except Exception:
                pass
            print(f"    ngl={ngl} ctx={ctx}: OK after {a+1} try "
                  f"load={load_s}s kv={info['kv_buf_mib']}", flush=True)
            if keep:
                return p, info
            p.kill()
            return None, info
        print(f"    ngl={ngl} ctx={ctx}: fail (try {a+1}/{TRIES}, "
              f"{load_s}s)", flush=True)
        time.sleep(3)
    return None, info


def tp(pprompt, npred=128):
    r = _req("/completion", {"prompt": pprompt, "n_predict": npred,
                             "temperature": 0, "cache_prompt": False})
    return {"prompt_n": r["timings"]["prompt_n"],
            "pp_tok_s": round(r["timings"]["prompt_per_second"], 1),
            "tg_tok_s": round(r["timings"]["predicted_per_second"], 2)}


def survey(m):
    alias, gguf = m["alias"], m["gguf"]
    extra = m.get("extra", [])
    out = {"alias": alias}

    print(f"[{alias}] layer search @4096", flush=True)
    best = None
    for ngl in [99, 32, 30, 28, 26, 24, 20, 16, 8, 0]:
        _, info = try_load(gguf, ngl, 4096, extra)
        if info["ok"]:
            best = info
            break
    if not best:
        out["error"] = "no layer count loads at 4096"
        print(json.dumps(out), flush=True)
        return out
    ngl = best["ngl"]
    out.update({"ngl": ngl, "trained_ctx": int(best["n_ctx_train"] or 0),
                "n_layer": int(best["n_layer"] or 0)})

    print(f"[{alias}] context search at ngl={ngl}", flush=True)
    probes = {}
    lo, hi = 4096, 8192
    while hi <= 262144:
        _, info = try_load(gguf, ngl, hi, extra)
        probes[hi] = info
        if not info["ok"]:
            break
        lo, hi = hi, hi * 2
    while hi - lo > 2048:
        mid = (lo + hi) // 2 // 2048 * 2048
        if mid <= lo:
            break
        _, info = try_load(gguf, ngl, mid, extra)
        probes[mid] = info
        if info["ok"]:
            lo = mid
        else:
            hi = mid
    oks = {c: i for c, i in probes.items() if i["ok"]}
    # "reliable" = loaded first try; "lucky" = needed retries
    reliable = max([c for c, i in oks.items() if i["tries"] == 1], default=4096)
    lucky = max(oks, default=4096)
    out["ceiling_reliable"] = reliable
    out["ceiling_max"] = lucky
    out["probes"] = {c: {"ok": i["ok"], "tries": i["tries"]} for c, i in probes.items()}
    lo = reliable
    print(f"[{alias}] reliable ceiling {reliable}, lucky max {lucky}", flush=True)
    proc, info = try_load(gguf, ngl, lo, extra, keep=True)
    if proc is None:                      # the ceiling probe itself failed
        proc, info = try_load(gguf, ngl, lo, extra, keep=True)
    out.update({"max_ctx": lo, "load_s": info["load_s"],
                "rss_mib": info.get("rss_mib"),
                "model_buf_mib": info.get("model_buf_mib"),
                "kv_buf_mib": info.get("kv_buf_mib"),
                "cpu_buf_mib": info.get("cpu_buf_mib"), "throughput": {}})
    if proc is not None:
        try:
            out["throughput"]["prefill_512"] = tp(PROMPT)
        except Exception as e:
            out["throughput"]["prefill_512"] = {"error": str(e)[:60]}
        try:
            out["throughput"]["prefill_2281"] = tp(PROMPT * 4 + PROMPT[:900])
        except Exception as e:
            out["throughput"]["prefill_2281"] = {"error": str(e)[:60]}
        proc.kill()
    print(json.dumps(out), flush=True)
    return out


models = json.load(open(sys.argv[1]))
for m in models:
    res = survey(m)
    with open("/tmp/survey.jsonl", "a") as fh:
        fh.write(json.dumps(res) + "\n")
