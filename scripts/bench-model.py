#!/usr/bin/env python3
"""Measure per-model load time, memory and tok/s on the Orin Nano via llama-server.

Usage: bench_one.py <alias> <gguf> <ngl> <ctx> [extra llama-server args...]
Prints one JSON object per model.
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

alias, model, ngl, ctx = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
extra = sys.argv[5:]

# The allocator fragments on this board: clear any stale server, force
# page-cache reclaim, and give the driver a moment before each load.
subprocess.run(["pkill", "-x", "llama-server"])
time.sleep(2)
subprocess.run(["python3", os.path.expanduser("~/bin/reclaim-free-pages.py"),
                "5000"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
time.sleep(3)

args = [BIN, "-m", model, "-ngl", ngl, "-c", ctx, "-fa", "on",
        "--cache-type-k", "q8_0", "--cache-type-v", "q8_0",
        "--no-mmap", "-np", "1", "--port", str(PORT),
        "--host", "127.0.0.1", "--jinja", "-lv", "4"] + extra
log = open(f"/tmp/bench-{alias}.log", "w")
t0 = time.time()
proc = subprocess.Popen(args, stdout=log, stderr=subprocess.STDOUT)


def get(path, data=None, timeout=1800):
    url = f"http://127.0.0.1:{PORT}{path}"
    body = json.dumps(data).encode() if data is not None else None
    req = urllib.request.Request(url, data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


# --- wait for load
load_s = None
while time.time() - t0 < 900:
    if proc.poll() is not None:
        print(json.dumps({"alias": alias, "error": "server died",
                          "rc": proc.returncode}))
        sys.exit(1)
    try:
        get("/health", timeout=3)
        load_s = time.time() - t0
        break
    except Exception:
        time.sleep(0.5)

props = get("/props")
rss_mib = int(subprocess.run(["ps", "-o", "rss=", "-p", str(proc.pid)],
                             capture_output=True, text=True).stdout.strip()) // 1024

para = ("The snow beach test corpus covers memory bandwidth, kernel launch "
        "overhead and cache behaviour on embedded GPUs. ") * 120
body = {"prompt": para, "n_predict": 128, "temperature": 0,
        "cache_prompt": False, "ignore_eos": True}

def run(b):
    t = time.time()
    r = get("/completion", b)
    wall = time.time() - t
    return {
        "prompt_n": r["timings"]["prompt_n"],
        "pp_tok_s": round(r["timings"]["prompt_per_second"], 2),
        "tg_tok_s": round(r["timings"]["predicted_per_second"], 2),
        "wall_s": round(wall, 2),
    }

runs = []
for _ in range(3):          # runs[0] is the cold prefill; the rest are warm
    runs.append(run(body))

# short prompt: latency-focused generation number
short = run({"prompt": "Write one sentence about the sea.", "n_predict": 64,
             "temperature": 0, "cache_prompt": False, "ignore_eos": True})

proc.kill()
log.close()
txt = open(f"/tmp/bench-{alias}.log").read()

def buf(name):
    m = re.search(rf"{name} buffer size\s*=\s*([\d.]+) MiB", txt)
    return round(float(m.group(1))) if m else None

def field(pat):
    m = re.search(pat, txt)
    return m.group(1).strip() if m else None

out = {
    "alias": alias,
    "ngl": int(ngl),
    "ctx": int(ctx),
    "load_s": round(load_s, 1),
    "rss_after_load_mib": rss_mib,
    "model_buf_mib": buf("CUDA0 model"),
    "kv_buf_mib": buf("CUDA0 KV"),
    "cpu_model_buf_mib": buf("CPU_Mapped model") or buf("CPU model"),
    "long_prompt": runs,
    "short_prompt": short,
    "n_ctx": props.get("default_generation_settings", {}).get("n_ctx"),
    "n_ctx_train": field(r"n_ctx_train\s*=\s*(\d+)"),
    "n_layer": field(r"n_layer\s*=\s*(\d+)"),
    "params": field(r"model params\s*=\s*([\d.]+ [BM])"),
    "ftype": field(r"file type\s*=\s*(\S+ \S+)"),
    "name": field(r"general.name str\s*=\s*(.+)"),
    "projected_mib": field(r"projected to use (\d+) MiB"),
}
print(json.dumps(out))
