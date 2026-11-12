#!/bin/bash
# Move every router GGUF off the Jetson's eMMC onto the NVMe, verified, behind symlinks.
#
# Why: eMMC (/dev/mmcblk0p1, /home) reads at ~47.7 MB/s; NVMe (/dev/nvme0n1p1) at
# ~2.5 GB/s. With --models-max 1 and load-mode = none every model pick is a full physical
# read, so a 3.5 GB model went from 48.8s to 10.4s cold after this.
#
# Safe to re-run: a path that is already the expected symlink is skipped, and the original
# is only deleted after both copies hash identically.
#
# Usage (on the Jetson):   ./move_models_to_nvme.sh [extra.gguf ...]

set -u
DST=/mnt/nvme/models/router
NSRC=/home/snowbeach/models/router
DSRC=/home/snowbeach/models/dory

log(){ echo "[$(date +%H:%M:%S)] $*"; }

mountpoint -q /mnt/nvme || { log "FATAL: /mnt/nvme not mounted (fstab has nofail, so this can silently be a dir on /)"; exit 1; }
mkdir -p "$DST"

files=(
  "$NSRC/deepseek-v4pro-9b.gguf"
  "$NSRC/hermes3-8b.gguf"
  "$NSRC/qwen35-4b-heretic.gguf"
  "$DSRC/NVIDIA-Nemotron-Labs-Dory1.0-4B-Q5_K_M.gguf"
)
[ "$#" -gt 0 ] && files=("$@")

for src in "${files[@]}"; do
  base=$(basename "$src")
  if [ -L "$src" ]; then log "SKIP $base (already a symlink)"; continue; fi
  [ -f "$src" ] || { log "SKIP $base (missing)"; continue; }

  log "COPY $base ($(du -h "$src" | cut -f1))"
  cp -f "$src" "$DST/$base" || { log "COPY FAILED $base"; continue; }

  a=$(sha256sum "$src" | cut -d' ' -f1)
  b=$(sha256sum "$DST/$base" | cut -d' ' -f1)
  if [ "$a" != "$b" ]; then
    log "HASH MISMATCH $base -- original left untouched, bad copy removed"
    rm -f "$DST/$base"; continue
  fi
  log "HASH OK $base"

  rm -f "$src" && ln -s "$DST/$base" "$src" && log "SYMLINKED $src -> $DST/$base"
done
log DONE  # presets unchanged; the next model pick resolves through the symlink
