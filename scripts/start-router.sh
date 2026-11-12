#!/bin/bash
# llama.cpp router on Snowbeach (Orin Nano 8GB): serves EVERY model in
# ~/models/router-presets.ini behind :8080, one resident at a time (--models-max 1).
#
#   hermes3-8b          -> Hermes-3-Llama-3.1-8B-lorablated Q4_K_M (ngl 99)
#   deepseek-v4pro-9b   -> DeepSeek-V4-Pro-Qwen3.5-9B-UNCENSORED-heretic Q4_K_M (ngl 28, ub 128)
#   qwen35-4b-heretic   -> Qwen3.5-4B heretic Q4_K_M (ngl 99, 32k ctx)
#   dory-4b             -> NVIDIA Nemotron-Labs Dory1.0-4B Q5_K_M, looped latent-recurrent
#                          (ngl 99, 2 loops, last_loop KV, reasoning off)
#
# Pick the model in the web UI's model selector, or pass "model" in the API body.
#
# BIN is the Dory-enabled llama.cpp fork (brianchmielNV/llama.cpp @ 98a74c9,
# ggml-org/llama.cpp#30306). It serves the other three models too. Stock mainline
# builds cannot load the dory architecture, and mainline is the fallback at
# ~/maple/llama.cpp/build/bin/llama-server (but its preset file needs `no-mmap`,
# not `load-mode`).

BIN=/home/snowbeach/maple/llama-dory/build-cuda/bin/llama-server
DIR=/home/snowbeach/models/router
LOG=/home/snowbeach/models/router.log
RECLAIM=$HOME/bin/reclaim-free-pages.py

pkill -x llama-server 2>/dev/null
sleep 3
fuser -k 8080/tcp 2>/dev/null
sleep 3

# contiguous cudaMalloc gets fragmented on a long-running board; force reclaim first
[ -f "$RECLAIM" ] && { python3 "$RECLAIM" 5000 >/dev/null 2>&1; sleep 2; }

setsid nohup "$BIN" \
  --models-dir "$DIR" \
  --models-preset /home/snowbeach/models/router-presets.ini \
  --models-max 1 \
  --host 0.0.0.0 --port 8080 \
  > "$LOG" 2>&1 < /dev/null &

# The router re-lists models as its children register, so poll until all four ids show up.
for i in $(seq 1 60); do
  sleep 2
  ids=$(curl -s http://127.0.0.1:8080/v1/models 2>/dev/null | grep -oE '"id":"[^"]+"' | cut -d'"' -f4 | tr '\n' ' ')
  case "$ids" in
    *hermes3-8b*dory-4b*|*dory-4b*hermes3-8b*)
      echo "router UP on :8080 — $ids"; exit 0 ;;
  esac
done
echo "router answered but did not expose the expected models; last list: ${ids:-<none>}"
echo "if only one id appears, a single-model systemd unit (hermes3-llama/deepseek-llama) is holding :8080 — see llama-router.service"
tail -20 "$LOG"
