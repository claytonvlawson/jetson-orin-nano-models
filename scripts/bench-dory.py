#!/usr/bin/env python3
"""Throughput for Dory on the dory llama.cpp fork (its own flags + override-kv).

Usage: bench-dory.py <gguf> <ngl> <ctx> [loops]
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

gguf, ngl, ctx = sys.argv[1], sys.argv[2], sys.argv[3]
loops = sys.argv[4] if len(sys.argv) > 4 else "2"

subprocess.run(["pkill", "-x", "llama-server"])
time.sleep(2)
subprocess.run(["python3", RECLAIM, "5000"], stdout=subprocess.DEVNULL,
               stderr=subprocess.DEVNULL)
time.sleep(3)

args = [BIN, "-m", gguf, "-ngl", ngl, "-c", ctx, "--flash-attn", "off",
        "--cache-type-k", "f16", "--cache-type-v", "f16",
        "--batch-size", "512", "--ubatch-size", "512",
        "--no-context-shift", "--load-mode", "none", "--jinja",
        "--reasoning", "on",
        "--override-kv",
        f"dory.num_loops=int:{loops},dory.recurrent_kv_cache_mode=str:per_loop",
        "-np", "1", "--port", str(PORT), "--host", "127.0.0.1", "-lv", "4"]
log = open("/tmp/dory-bench.log", "w")
t0 = time.time()
p = subprocess.Popen(args, stdout=log, stderr=subprocess.STDOUT)


def get(path, body=None, timeout=1800):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}{path}", data=data,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


load_s = None
while time.time() - t0 < 900:
    if p.poll() is not None:
        print(json.dumps({"error": "server died"}))
        sys.exit(1)
    try:
        get("/health", timeout=3)
        load_s = round(time.time() - t0, 1)
        break
    except Exception:
        time.sleep(0.5)

rss = int(subprocess.run(["ps", "-o", "rss=", "-p", str(p.pid)],
                         capture_output=True, text=True).stdout.strip()) // 1024
para = ("The snow beach test corpus covers memory bandwidth, kernel launch "
        "overhead and cache behaviour on embedded GPUs. ") * 27


def run(prompt, npred=128):
    r = get("/completion", {"prompt": prompt, "n_predict": npred,
                            "temperature": 0, "cache_prompt": False,
                            "ignore_eos": True})
    t = r["timings"]
    return {"prompt_n": t["prompt_n"],
            "pp_tok_s": round(t["prompt_per_second"], 1),
            "tg_tok_s": round(t["predicted_per_second"], 2)}


res = {"alias": "dory1.0-4b-q5", "ngl": int(ngl), "ctx": int(ctx),
       "loops": loops, "load_s": load_s, "rss_mib": rss, "runs": []}
for _ in range(3):
    res["runs"].append(run(para))
res["long"] = run(para * 4 + para[:900])
p.kill()
log.close()
txt = open("/tmp/dory-bench.log", errors="ignore").read().replace("\x00", "")
for k, pat in (("model_buf_mib", r"CUDA0 model buffer size\s*=\s*([\d.]+)"),
               ("kv_buf_mib", r"CUDA0 KV buffer size\s*=\s*([\d.]+)"),
               ("cpu_buf_mib", r"CUDA_Host model buffer size\s*=\s*([\d.]+)")):
    m = re.search(pat, txt)
    res[k] = m.group(1) if m else None
print(json.dumps(res))
