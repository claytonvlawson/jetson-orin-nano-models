# Changelog

Versioned changes to the Jetson Orin Nano local model setup. Numbers are measured on
`snowbeach` (Orin Nano Super, 8 GB, JetPack R36.4.3, CUDA 12.6, `MAXN_SUPER`).

## v1.1.0 — 2026-11-15

Autocompact: long chats now fold instead of failing, and the tunnel stops dying.

### Added

- **`scripts/context_guard.py`** — the autocompact proxy that was running on the
  board but not in this repo. It listens on `127.0.0.1:8090`, counts each request
  with the model's own `POST /tokenize`, and when the prompt crosses the trigger it
  folds the older turns into a markdown checkpoint at
  `~/contexts/compact-YYYYmmdd-HHMMSS.md`, rewrites the request as system +
  compacted block + the last `keep_recent_messages` turns, and streams the reply
  back. Everything else passes through untouched.
- **`trigger_tokens`** — an absolute compaction threshold. The guard could only
  trigger as a fraction of the window (`trigger_ratio`, 0.80 of 32768 = 26214),
  which is too close to the memory cliff to be useful. `trigger_tokens` wins over
  the ratio when greater than zero; set to **16000** here. New helper
  `trigger_tokens_for(n_ctx)` is the single place the threshold is computed, and
  both `/guard/status` and the startup log report it.
- **`scripts/guard-config.json`** — the live guard config, including
  `trigger_tokens: 16000`.
- **`scripts/jetson-tunnel.sh`** — Mac-side tunnel that probes the LAN address
  (`10.0.0.13:22`) first and falls back to the tailnet IP, then execs ssh with
  `-L 127.0.0.1:8080:127.0.0.1:8090`. launchd runs it with `KeepAlive`, so every
  restart re-picks the best path.
- **§5 in the README** — the autocompact design, the trigger, the two failure modes
  that cost a debugging session, and the verification commands.

### Fixed

- **The guard was not in the path.** The Mac tunnel forwarded
  `127.0.0.1:8080 -> jetson:8080`, i.e. straight to the router, bypassing the guard
  on 8090. That is why a 33,581-token chat returned
  `request (33581 tokens) exceeds the available context size (32768 tokens)`
  instead of compacting. The tunnel now lands on 8090.
- **`Accept-Encoding` was dropped upstream.** `_forward()` passed only
  `Content-Type`, `Authorization` and `Accept`. llama-server serves its web UI
  gzip-only and answers `415 Error: gzip is not supported by this browser` to
  clients that do not advertise gzip, so every static asset and `/sw.js` 415'd
  behind the guard and the browser UI broke. The header (and `User-Agent`) is now
  forwarded; response headers were already relayed verbatim, so `Content-Encoding`
  survives the round trip.
- **The tunnel no longer pins the tailnet IP.** The Mac↔Jetson Tailscale path drops
  every few minutes (tailscaled churns the peer endpoint between
  `10.0.0.181:41641` and `[fe80::…%en0]:41641`). A tunnel pinned to
  `100.72.223.15` dies mid-request: the UI shows "Server unavailable" and long
  completions get `ConnectionResetError`. `tailscale down && tailscale up` on the
  Mac clears it temporarily; the LAN-first script is the durable fix.

### Verified (2026-11-15)

    T1 status        trigger_tokens 16000, n_ctx 32768, output_dir ~/contexts   PASS
    T2 forced fold   3745 -> 2711 tokens, 31.5s, reply "The trigger threshold
                     is 16,000 tokens.", file compact-20261115-142944.md         PASS
    T3 passthrough   X-Context-Guard: passthrough, 1.2s, no rewrite              PASS
    T4 web UI        / 200, /sw.js 200, /v1/models 200, /props 200 via :8080     PASS
    T5 big fold      39079 -> 11193 tokens, 13 messages folded, 122 KB handoff   PASS

### Known hazard

- **Do not test compaction with a 30k+ payload on this board.** The 39,079-token
  test folded correctly and wrote its handoff, then the board went into a swap
  storm: ping still answered, sshd accepted TCP, but the banner exchange timed out
  and six retries over two minutes failed. Power cycle required. Real chats compact
  at 16000, well below that cliff. Verify with a ~3k history plus
  `X-Force-Compact: 1`.

## v1.0.0 — 2026-11-12

Storage and measurement baseline.

- Moved the served GGUFs to the NVMe behind symlinks: cold loads went from 48.8 s to
  10.4 s (microSD 47.7 MB/s vs NVMe 2.5 GB/s). `scripts/move_models_to_nvme.sh`
  copies, sha256-verifies, then leaves the symlink.
- Measured specs for all four models, including Dory1.0-4B at two loops.
- Per-model `ngl` ladders, the page-reclaim step, and the context ceilings from the
  sweep.
