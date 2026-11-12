#!/usr/bin/env python3
"""Same survey as survey-ceiling.py but for the Dory build.

Dory needs a llama.cpp fork (ggml-org/llama.cpp#30306) and its own flags: the
card pins --flash-attn off, f16 KV, ub 512, and a dory.num_loops override.
Stock llama-server cannot load this GGUF at all.
"""
import json
import os
import re
import subprocess
import sys
import time
import urllib.request

BIN = "/home/snowbeach/maple/llama-dory/build-cuda/bin/llama-server"
PORT = 8099
RECLAIM = os.path.expanduser("~/bin/reclaim-free-pages.py")
TRIES = 3
PROMPT = ("The snow beach test corpus covers memory bandwidth, kernel launch "
          "overhead and cache behaviour on embedded GPUs. ") * 27
OVERRIDE = "dory.num_loops=int:2,dory.recurrent_kv_cache_mode=str:per_loop"


def memfree_mb():
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemFree"):
                return int(line.split()[1]) // 1024
    return 0


def clear():
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


def load_once(model, ngl, ctx):
    clear()
    args = [BIN, "-m", model, "-ngl", str(ngl), "-c", str(ctx),
            "--flash-attn", "off",
            "--cache-type-k", "f16", "--cache-type-v", "f16",
            "--batch-size", "512", "--ubatch-size", "512",
            "--no-context-shift", "--load-mode", "none", "--jinja",
            "--reasoning", "on",
            "--override-kv", OVERRIDE,
            "-np", "1", "--port", str(PORT), "--host", "127.0.0.1", "-lv", "4"]
    lp = f"/tmp/dory-n{ngl}-c{ctx}.log"
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


def try_load(model, ngl, ctx, keep=False):
    info = {"ngl": ngl, "ctx": ctx, "ok": False, "tries": 0}
    for a in range(TRIES):
        info["tries"] = a + 1
        p, ok, load_s, lp = load_once(model, ngl, ctx)
        info["load_s"] = load_s
        if ok:
            txt = open(lp, errors="ignore").read()

            def f(pat):
                m = re.search(pat, txt)
                return m.group(1) if m else None
            info.update({"ok": True,
                         "model_buf_mib": f(r"CUDA0 model buffer size\s*=\s*([\d.]+)"),
                         "kv_buf_mib": f(r"CUDA0 KV buffer size\s*=\s*([\d.]+)"),
                         "cpu_buf_mib": f(r"CUDA_Host model buffer size\s*=\s*([\d.]+)"),
                         "n_ctx_train": f(r"n_ctx_train\s*=\s*(\d+)"),
                         "n_layer": f(r"n_layer\s*=\s*(\d+)")})
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


def tp(prompt, npred=128):
    r = _req("/completion", {"prompt": prompt, "n_predict": npred,
                             "temperature": 0, "cache_prompt": False,
                             "ignore_eos": True})
    return {"prompt_n": r["timings"]["prompt_n"],
            "pp_tok_s": round(r["timings"]["prompt_per_second"], 1),
            "tg_tok_s": round(r["timings"]["predicted_per_second"], 2)}


model = sys.argv[1]
out = {"alias": "dory1.0-4b-q5"}

print("[dory] layer search @4096", flush=True)
best = None
for ngl in [99, 32, 28, 24, 20, 16]:
    _, info = try_load(model, ngl, 4096)
    if info["ok"]:
        best = info
        break
if not best:
    out["error"] = "does not load"
    print(json.dumps(out))
    sys.exit(1)
ngl = best["ngl"]
out.update({"ngl": ngl, "trained_ctx": int(best["n_ctx_train"] or 0),
            "n_layer": int(best["n_layer"] or 0)})

print(f"[dory] context search at ngl={ngl}", flush=True)
probes = {}
lo, hi = 4096, 8192
while hi <= 262144:
    _, info = try_load(model, ngl, hi)
    probes[hi] = info
    if not info["ok"]:
        break
    lo, hi = hi, hi * 2
while hi - lo > 2048:
    mid = (lo + hi) // 2 // 2048 * 2048
    if mid <= lo:
        break
    _, info = try_load(model, ngl, mid)
    probes[mid] = info
    if info["ok"]:
        lo = mid
    else:
        hi = mid
oks = {c: i for c, i in probes.items() if i["ok"]}
reliable = max([c for c, i in oks.items() if i["tries"] == 1], default=4096)
out.update({"ceiling_reliable": reliable, "ceiling_max": max(oks, default=4096),
            "probes": {c: {"ok": i["ok"], "tries": i["tries"]}
                       for c, i in probes.items()}})

proc, info = try_load(model, ngl, reliable, keep=True)
out.update({"max_ctx": reliable, "load_s": info["load_s"],
            "rss_mib": info.get("rss_mib"),
            "model_buf_mib": info.get("model_buf_mib"),
            "kv_buf_mib": info.get("kv_buf_mib"),
            "cpu_buf_mib": info.get("cpu_buf_mib"), "throughput": {}})
if proc is not None:
    for k, pr in (("prefill_512", PROMPT),
                  ("prefill_2281", PROMPT * 4 + PROMPT[:900])):
        try:
            out["throughput"][k] = tp(pr)
        except Exception as e:
            out["throughput"][k] = {"error": str(e)[:60]}
    proc.kill()
print(json.dumps(out), flush=True)
with open("/tmp/dory.json", "w") as fh:
    fh.write(json.dumps(out))
