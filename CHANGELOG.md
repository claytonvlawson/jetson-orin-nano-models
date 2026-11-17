# Changelog

Versioned changes to the Jetson Orin Nano local model setup. Numbers are measured on
`snowbeach` (Orin Nano Super, 8 GB, JetPack R36.4.3, CUDA 12.6, `MAXN_SUPER`).

## v1.2.0 — 2026-11-17

Full re-verification of every number against the board. No new features; this release
is the correction pass, and several claims in v1.0.0 and v1.1.0 did not survive it.

### Fixed (numbers)

- **Storage throughput was wrong in both directions.** v1.0.0 claimed microSD
  47.7 MB/s against NVMe 2.5 GB/s (52x). Re-measured with `dd iflag=direct`, 1 GiB
  reads: microSD **86.5 MB/s** read / 62.6 MB/s write, NVMe **1.7 GB/s** read /
  592 MB/s write. The real ratio is ~20x. The disk table is refreshed from live `df`.
- **Layer counts did not add up.** The README said hermes3-8b was "33/34", qwen35-4b
  "33/33", and the 9B "32 transformer layers" followed two lines later by "34/34".
  Block counts read out of the GGUF headers: hermes3-8b 32 (llama), the three qwen35
  models 33, Dory 72 with `num_loops=4` in the file. With the output layer that is
  33 or 34 tensors, and the tables now say which.
- **Dory's "113 layers" was never captured.** No log on the board contains a runtime
  layer count for Dory, and the GGUF says 72 blocks. The claim is withdrawn and
  flagged as unverified rather than replaced with a guess.
- **The served model set had changed and the README had not.** `deepseek-v4pro-9b-q2`
  is gone from the NVMe and from the router; `deepseek-v4pro-4b-heretic` was added on
  2026-11-15. The models table, the fit table, the ceilings table and the preset block
  now list the five aliases that actually answer.
- **The §4 preset block was three revisions stale.** It showed `load-mode = none`,
  the 9B at `ngl 99`, qwen at 16384, a q2 alias, and no 4B. Replaced with the file the
  board runs, and `scripts/router-presets.ini` in the repo is now that same file.

### Added (measurements)

- **Cold swap, end to end, 2026-11-17**, five aliases: deepseek-v4pro-4b-heretic 6.0 s,
  qwen35-4b-heretic 6.6 s, hermes3-8b 6.6 s, deepseek-v4pro-9b 9.4 s, dory-4b 9.8 s.
  All five loaded on the first attempt. The pre-move card column is kept beside it
  (48–71 s).
- **A note that the ceilings table is a floor, not a maximum.** The sweep ran under
  `load-mode = none`; under `dio` the 4B models serve 32768 first try, well past the
  16384 the sweep called reliable. Re-sweeping under `dio` is listed as open work.

### Known gaps carried forward

- `deepseek-v4pro-4b-heretic` has a cold-swap number but no throughput bench.
- `llama-cache-guard.timer` runs on the board and is not in this repo.

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
