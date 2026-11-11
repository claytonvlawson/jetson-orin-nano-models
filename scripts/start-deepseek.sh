#!/bin/bash
# Serve DeepSeek-V4-Pro-Qwen3.5-9B on Snowbeach (Orin Nano 8GB).
#   ./start.sh        -> Q4_K_M, best quality, layers tuned to the 3.9 GB nvmap cap
#   ./start.sh q2     -> Q2_K, fully offloaded, ~10 tok/s (lower quality)
#
# Measured decode on this board (ctx 4096, q8_0 KV, MAXN_SUPER, np 1):
#   Q4_K_M ngl 20 ub512  6.63 tok/s   (old default, 4 slots)
#   Q4_K_M ngl 24 ub128  7.30-7.45
#   Q4_K_M ngl 28 ub128  8.20-8.28   <- served by default (after a page reclaim)
#   Q4_K_M ngl 99        fails: 5.3 GB > ~3.9 GB single cudaMalloc (nvmap 32-bit)
#   Q2_K   ngl 99 ub512  9.97 tok/s   <- fastest, ./start.sh q2
# -ub 128 shrinks the CUDA compute buffers, which is what lets ngl 24-28 fit.
# -np 1 instead of the default 4 slots frees per-slot KV buffers.

DIR=/home/snowbeach/models/deepseek-v4pro-9b
BIN=/home/snowbeach/maple/llama.cpp/build/bin/llama-server
LOG=$DIR/server.log
RECLAIM=$HOME/bin/reclaim-free-pages.py
Q4=$DIR/DeepSeek-V4-Pro-Qwen3.5-9B-UNCENSORED-heretic.i1-Q4_K_M.gguf
Q2=$DIR/DeepSeek-V4-Pro-Qwen3.5-9B-UNCENSORED-heretic.i1-Q2_K.gguf

pkill -x llama-server 2>/dev/null
sleep 3
fuser -k 8080/tcp 2>/dev/null
sleep 3

# ngl 28 needs a contiguous cudaMalloc that a long-running board no longer has.
# Touch anonymous pages to force reclaim before the launch attempts.
if [ -f "$RECLAIM" ]; then
  python3 "$RECLAIM" 2000 >/dev/null 2>&1
  sleep 2
fi

start() { # $1=model $2=ngl $3=ctx $4=ub $5=label
  local alog="$DIR/attempt-ngl$2.log"
  [ "$2" != "0" ] && local extra="-fa on -ctk q8_0 -ctv q8_0 -ub $4" || local extra=""
  setsid nohup "$BIN" -m "$1" -c "$3" -ngl "$2" -np 1 $extra \
    --no-mmap --host 0.0.0.0 --port 8080 --jinja \
    --alias deepseek-v4pro-9b > "$alog" 2>&1 < /dev/null &
  for i in $(seq 1 200); do
    sleep 2
    grep -q "listening on http" "$alog" 2>/dev/null && { cp "$alog" "$LOG"; echo "UP: $5 (ngl $2, ctx $3, ub $4)"; return 0; }
    grep -q "exiting due to model loading error" "$alog" 2>/dev/null && {
      echo "FAILED: $5 ngl $2 :: $(grep -hoE 'allocating [0-9.]+ MiB|failed to allocate buffer for [a-z ]+' "$alog" | tail -1)"
      pkill -x llama-server; sleep 3; return 1; }
  done
  pkill -x llama-server; sleep 3; echo "TIMEOUT: $5 ngl $2"; return 1
}

if [ "$1" = "q2" ]; then
  start "$Q2" 99 4096 512 "Q2_K full offload" && exit 0
  echo "Q2_K failed, falling back to Q4_K_M"
fi

start "$Q4" 28 4096 128 "Q4_K_M" \
  || start "$Q4" 26 4096 128 "Q4_K_M" \
  || start "$Q4" 24 4096 128 "Q4_K_M" \
  || start "$Q4" 20 4096 512 "Q4_K_M" \
  || { echo "GPU paths failed, CPU only"; start "$Q4" 0 2048 512 "Q4_K_M CPU"; }
