#!/bin/bash
# jetson-tunnel.sh - keep 127.0.0.1:8080 forwarded to the Jetson's context guard.
#
# Why this exists: the Tailscale path between this Mac and the Jetson drops every
# few minutes (tailscaled churns the peer endpoint between 10.0.0.181:41641 and
# [fe80::...%en0]:41641). When it drops, an ssh tunnel pinned to the tailnet IP
# dies mid-request and the :8080 web UI reports "Server unavailable".
#
# So: prefer the LAN address (stable, ~5ms) and fall back to the tailnet IP when
# the Mac is away from home. launchd restarts this on exit, so every reconnect
# re-picks the best path.
#
# Local 8080 -> Jetson 8090 (llama-context-guard) -> Jetson 8080 (llama-router).

set -u

LAN_HOST="10.0.0.13"
TS_HOST="100.72.223.15"
REMOTE_PORT="8090"
LOCAL_PORT="8080"
USER_NAME="snowbeach"
KEY="$HOME/.ssh/id_ed25519"

pick_host() {
    local h
    for h in "$LAN_HOST" "$TS_HOST"; do
        if nc -z -G 2 "$h" 22 >/dev/null 2>&1; then
            printf '%s' "$h"
            return 0
        fi
    done
    return 1
}

HOST="$(pick_host)" || HOST="$TS_HOST"
echo "$(date '+%Y-%m-%d %H:%M:%S') tunnel -> ${HOST}:22 (local ${LOCAL_PORT} -> remote ${REMOTE_PORT})"

exec /usr/bin/ssh -N -T \
    -o ExitOnForwardFailure=yes \
    -o ServerAliveInterval=15 \
    -o ServerAliveCountMax=3 \
    -o ConnectTimeout=10 \
    -o BatchMode=yes \
    -o IdentitiesOnly=yes \
    -o StrictHostKeyChecking=accept-new \
    -i "$KEY" \
    -l "$USER_NAME" \
    -L "127.0.0.1:${LOCAL_PORT}:127.0.0.1:${REMOTE_PORT}" \
    "$HOST"
