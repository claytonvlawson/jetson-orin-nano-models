# Jetson Orin Nano Super  *Local Model Setup

How five GGUF models were added to a Jetson Orin Nano Super Developer Kit (8 GB
unified memory), served behind one `llama.cpp` router, and kept inside the memory
budget. Everything here was run on the board; numbers are measured, not estimated.

Board: `snowbeach` — Jetson Orin Nano Super Developer Kit, aarch64, Ubuntu 22.04,
JetPack R36.4.3, CUDA 12.6, 8 GB unified LPDDR5, power mode `MAXN_SUPER`.

**Version 1.2.0** — every number re-verified against the board on 2026-11-17: NVMe
storage and cold-swap figures, the five served aliases, GGUF-read layer counts, and the
`dio` load mode. See [CHANGELOG.md](CHANGELOG.md).

Measured on llama.cpp build `10291 (7e30f3adb)` for the Qwen and DeepSeek builds,
and on the Dory fork build `1 (98a74c9)` for Dory — see
[Measured performance](#measured-performance).

## Storage

The kit boots from the 512 GB microSD card; the NVMe drive is secondary storage.
Both are in use — the weights that matter are on the NVMe.

| Device | Size | Mount | Used | Free |
|---|---|---|---|---|
| `/dev/mmcblk0p1` (512 GB microSD) | 468 G | `/` | 267 G | 182 G |
| `/dev/nvme0n1p1` (NVMe SSD) | 234 G | `/mnt/nvme` | 137 G | 85 G |

Measured 2026-11-17. The NVMe in this build is 256 GB class, not 8 GB — the "8 GB" in
the kit name is the unified RAM. Keep that straight: 8 GB is the hard ceiling on what
can be resident, and storage is not the constraint.

The served GGUFs live on the NVMe and are symlinked back into `~/models/router/`, so
every preset path still resolves. The move was 2026-11-12 for the four original models
and 2026-11-15 for the DeepSeek 4B. Downloads and non-served files stay on `/home`
(microSD). The original argument for keeping weights on the card was that a card swap
does not lose them; [Cold loads](#cold-loads-why-the-ggufs-moved-to-nvme) is why that
tradeoff was reversed. A card swap no longer protects the model files, and an NVMe
failure now takes them with it; keeping a copy elsewhere is on you.

### Cold loads: why the GGUFs moved to NVMe

Load time on a router is `file size / disk read speed`. Re-measured 2026-11-17 with
`dd ... iflag=direct` (page cache bypassed, 1 GiB reads):

| Device | Sequential read | Sequential write |
|---|---|---|
| microSD (`/`, 512 GB card) | 86.5 MB/s | 62.6 MB/s |
| NVMe (`/mnt/nvme`) | 1.7 GB/s | 592 MB/s |

That is ~20x on reads. The first version of this section claimed 47.7 MB/s against
2.5 GB/s (52x) — both figures were measured under different conditions and neither
reproduces with a direct read of the actual files. Use the table above.

With `--models-max 1` every model pick is a straight read of the whole file, so the
disk, not the GPU, sets the switch latency. Cold swap, measured end to end on
2026-11-17 (`load-mode = dio`, board otherwise idle, load + prefill + 8 tokens):

| Alias | File | Cold swap |
|---|---|---|
| `deepseek-v4pro-4b-heretic` | 2.78 GB | **6.0 s** |
| `qwen35-4b-heretic` | 2.71 GB | **6.6 s** |
| `hermes3-8b` | 4.92 GB | **6.6 s** |
| `deepseek-v4pro-9b` | 5.63 GB | **9.4 s** |
| `dory-4b` | 3.67 GB | **9.8 s** |

Pre-move, on the card: 48.8 s for Dory (measured), 71 s for the 9B and 65 s for the
8B (from the earlier README, not re-measured). The pre-move column survives in
[Measured performance](#measured-performance) for comparison.

The cutover is a symlink swap, so nothing but the bytes moved: presets keep their
paths, the systemd unit is untouched, no restart or root needed, and an already-running
child is unaffected because it holds an open fd. `scripts/move_models_to_nvme.sh` does it
and is safe to re-run — it copies, verifies sha256 on both copies, and only then deletes
the original and links:

```
~/models/router/<name>.gguf  ->  /mnt/nvme/models/router/<name>.gguf
~/models/dory/Dory1.0-4B...gguf -> /mnt/nvme/models/router/NVIDIA-Nemotron-Labs-Dory1.0-4B-Q5_K_M.gguf
```

`/mnt/nvme` is in `/etc/fstab` with `nofail`, so the links survive a reboot; a failed NVMe
mount leaves them dangling (the mount is `generated` per `systemctl is-enabled`, i.e.
fstab owns it — do not symlink before checking that).

Rollback: `cp` the file back out of `/mnt/nvme` and replace the symlink with it.

## The Models

Five GGUFs, five aliases, one router. Sizes are GB (10^9); block counts and context
lengths are read from the GGUF headers with a stdlib parser, not from model cards.

| Alias | File | Quant | Size | Blocks | Trained ctx | Source |
|---|---|---|---|---|---|---|
| `hermes3-8b` | `hermes3-8b.gguf` | Q4_K_M | 4.92 GB | 32 (llama) | 131072 | [bartowski/Hermes-3-Llama-3.1-8B-lorablated-GGUF](https://huggingface.co/bartowski/Hermes-3-Llama-3.1-8B-lorablated-GGUF) |
| `deepseek-v4pro-9b` | `deepseek-v4pro-9b.gguf` | i1 Q4_K_M | 5.63 GB | 33 (qwen35) | 262144 | [mradermacher/DeepSeek-V4-Pro-Qwen3.5-9B-UNCENSORED-heretic-i1-GGUF](https://huggingface.co/mradermacher/DeepSeek-V4-Pro-Qwen3.5-9B-UNCENSORED-heretic-i1-GGUF) |
| `deepseek-v4pro-4b-heretic` | `DeepSeek-V4-Pro-Qwen3.5-4B-UNCENSORED-heretic.Q4_K_M.gguf` | Q4_K_M | 2.78 GB | 33 (qwen35) | 262144 | [mradermacher](https://huggingface.co/mradermacher) DeepSeek-V4-Pro-Qwen3.5-4B-UNCENSORED-heretic |
| `qwen35-4b-heretic` | `qwen35-4b-heretic.gguf` | Q4_K_M | 2.71 GB | 33 (qwen35) | 262144 | [mradermacher/Qwen3.5-4B-heretic-GGUF](https://huggingface.co/mradermacher/Qwen3.5-4B-heretic-GGUF) |
| `dory-4b` | `NVIDIA-Nemotron-Labs-Dory1.0-4B-Q5_K_M.gguf` | Q5_K_M | 3.67 GB | 72 (dory), `num_loops=4` in file | 262144 | [nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-GGUF](https://huggingface.co/nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-GGUF) |

A sixth alias, `deepseek-v4pro-9b-q2`, is **retired**. It reached 77824 context on this
board (the only model here that does) but the Q2 GGUF is no longer on the NVMe and the
alias is not served. Its measurements stay in `results/q2-bench.json` as history.

What Each One Is:

- **hermes3-8b** — Hermes 3 Llama 3.1 8B, loRA-ablated, then imatrix-quantized by
  bartowski. 32 blocks, so 33 tensors with the output layer. The most reliable general
  chat model here and the one with the least room to spare: `ngl 99` (4403 MiB model
  buffer) loaded for the throughput runs but failed 3/3 in the ceiling sweep, while
  `ngl 32` (4271 MiB, 32 of 33 tensors on the GPU) loaded first try every time. The
  132 MiB difference is the output layer, and on an 8 GB board that is the margin.
- **Deepseek-v4pro-9b (Q4)** — DeepSeek-V4-Pro distilled into Qwen3.5-9B, then
  "heretic" refusal-ablated (ARA). 8.95 B params, 33 blocks (34 tensors with the output
  layer), hybrid linear+full attention, native 262K context. The strongest reasoner
  here. Earlier notes claimed it could only offload 28 of 32 layers; with the
  page-reclaim fix in §3a it offloads all 34/34 at `ngl 99`, and layers are not what
  limits it — context is. The served preset still runs `ngl 28` because at 4096 context
  the extra layers buy nothing measurable and cost the CUDA allocator headroom.
- **Deepseek-v4pro-4b-heretic** — the same DeepSeek-V4-Pro distill at 4B, heretic-
  ablated, Q4_K_M, 2.59 GiB. 33 blocks. Takes after `qwen35-4b-heretic`: `ngl 99` and
  32K context both fit, which makes it the cheap reasoning tier that does not cost the
  big model's headroom.
- **Qwen35-4b-heretic** — Qwen3.5-4B, language tower ablated (vision encoder left
  alone). 4 B params, 33 blocks, 262K trained context. The cheap fast one, and the best
  of the five for long inputs that fit inside 32K.
- **Dory** — NVIDIA Nemotron Labs Dory1.0-4B, a *looped* (latent-recurrent)
  normalized-GPT: the shared recurrent block is applied N times at inference. The GGUF
  carries 72 blocks and `num_loops=4`; the board runs it at 2 loops through
  `dory.num_loops` overrides, and `last_loop` keeps the KV cache at one-loop size. Needs
  its own llama.cpp fork (§1b). The most token-efficient per parameter and the slowest
  per token. **An earlier version of this README said "113 layers, 99 offloaded"; the
  GGUF says 72 blocks, and the runtime count was never captured in a log, so treat any
  layer count for Dory as unverified.**

## 1. Build llama.cpp with CUDA for Orin (sm_87)

Stock llama.cpp ships no CUDA kernels for Orin's compute capability, so the build
has to name it explicitly. A stale CMake cache silently keeps the old arch, which
is why the cache is wiped first.

```bash
cd ~
git clone https://github.com/ggml-org/llama.cpp maple/llama.cpp
cd maple/llama.cpp

rm -rf build
cmake -B build -DGGML_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=87 \
  -DGGML_CUDA_FA=ON \
  -DLLAMA_CURL=ON
cmake --build build --config Release -j"$(nproc)"

./build/bin/llama-server --version
# version: 10291 (7e30f3adb) — built with GNU 11.4.0 for Linux aarch64
```

Verified cache settings on this board:

```
CMAKE_CUDA_ARCHITECTURES = 87
GGML_CUDA                = ON
GGML_CUDA_FA             = ON
GGML_CUDA_GRAPHS         = ON
```

`GGML_CUDA_FA=ON` matters more than it looks — without flash attention the KV cache
for a 9B model at usable context does not fit in 8 GB.

### 1b. Dory needs a second build

Dory's looped architecture is not in a llama.cpp release yet
([ggml-org/llama.cpp#30306](https://github.com/ggml-org/llama.cpp/pull/30306)).
Until it lands, build the pinned fork revision next to the stock tree:

```bash
cd ~
git clone https://github.com/ggml-org/llama.cpp maple/llama-dory
cd maple/llama-dory
git checkout 98a74c97ea09ac72f6ceda1559808fc0a2f447ce

cmake -B build-cuda -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=87 -DLLAMA_CURL=ON
cmake --build build-cuda --config Release --target llama-server -j2

./build-cuda/bin/llama-server --version
# version: 0.6.0-dev (build 1, commit 98a74c9)
```

`-j2` is deliberate. With `-j"$(nproc)"` the CUDA template-instance compile peaks
over 4 GB of RSS on an 8 GB board and the kernel OOM-killer starts hunting compilers.

The stock `llama-server` cannot load the Dory GGUF at all — it exits in under a
second with `failed to load model`. That failure mode means "wrong binary", not
"broken file".

On this board the router itself runs the fork binary (`maple/llama-dory/build-cuda/bin/llama-server`),
so all five models — Dory included — are served by one process on one port. The fork
also dropped `--no-mmap`: the replacement is `load-mode` in the preset file, which this
board runs as `dio` (see §4 for why, and for the version that used `none`).

## 2. Pull the GGUFs

`hf` is not installed on the board by default; install the CLI once, then download
only the single quant you need. Do not snapshot the whole repo — the F16 and Q8
files alone are 18 GB and 9.5 GB.

```bash
pip3 install --user -U "huggingface_hub[cli]"
export PATH="$HOME/.local/bin:$PATH"

# 8B chat model (~4.9 GB)
hf download bartowski/Hermes-3-Llama-3.1-8B-lorablated-GGUF \
  Hermes-3-Llama-3.1-8B-lorablated-Q4_K_M.gguf \
  --local-dir ~/models/Hermes-3-Llama-3.1-8B-lorablated-GGUF

# 9B reasoner (~5.6 GB). i1-Q4_K_M; i1-Q2_K is the 3.8 GB fallback.
hf download mradermacher/DeepSeek-V4-Pro-Qwen3.5-9B-UNCENSORED-heretic-i1-GGUF \
  DeepSeek-V4-Pro-Qwen3.5-9B-UNCENSORED-heretic.i1-Q4_K_M.gguf \
  --local-dir ~/models/deepseek-v4pro-9b

# 4B heretic (~2.7 GB)
hf download mradermacher/Qwen3.5-4B-heretic-GGUF \
  Qwen3.5-4B-heretic.Q4_K_M.gguf \
  --local-dir ~/models/qwen35-4b-heretic

# Dory (~3.67 GB). Needs the Dory build from §1b, not the stock one.
hf download nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-GGUF \
  NVIDIA-Nemotron-Labs-Dory1.0-4B-Q5_K_M.gguf \
  --local-dir ~/models/dory
```

Then stage flat copies in the router directory, because the router keys models by
the filename in its `--models-dir`:

```bash
mkdir -p ~/models/router
cp ~/models/Hermes-3-Llama-3.1-8B-lorablated-GGUF/Hermes-3-Llama-3.1-8B-lorablated-Q4_K_M.gguf ~/models/router/hermes3-8b.gguf
cp ~/models/deepseek-v4pro-9b/DeepSeek-V4-Pro-Qwen3.5-9B-UNCENSORED-heretic.i1-Q4_K_M.gguf ~/models/router/deepseek-v4pro-9b.gguf
cp ~/models/qwen35-4b-heretic/Qwen3.5-4B-heretic.Q4_K_M.gguf ~/models/router/qwen35-4b-heretic.gguf
cp ~/models/dory/NVIDIA-Nemotron-Labs-Dory1.0-4B-Q5_K_M.gguf ~/models/router/dory.gguf
```

The 9B quant name in this repo follows mradermacher's `.i1-Q4_K_M` suffix, while the
4B and 8B use plain `-Q4_K_M`. Check the actual filename in the repo's file list
before downloading; renaming after download is fine.

## 3. Tune for 8 GB

Three constraints drive every flag:

1. **8 GB is shared** between CPU and GPU. A 5.6 GB model plus a KV cache plus
   CUDA context overhead does not leave room for much context.
2. **Long-running boards fragment the CUDA allocator.** A multi-GB `cudaMalloc`
   fails on a board that has been up for days even when `free` shows room. See §3a —
   the reclaim target in the original notes was too small.
3. **KV cache is the second-largest allocation.** Quantizing it to `q8_0` roughly
   halves it versus f16 and is the difference between fitting and not.

### What Fit, Per Model

These are settings that load, not theoretical maxima. Measured maxima are in
[Ceilings](#ceilings-how-far-each-model-actually-goes).

| Alias | GPU layers | Context | KV cache | Extra |
|---|---|---|---|---|
| `hermes3-8b` | 32 of 33 | 4096 | q8_0 | — |
| `deepseek-v4pro-9b` | 28 of 34 | 4096 | q8_0 | `ub 128` |
| `deepseek-v4pro-4b-heretic` | 99 (all 34/34) | 32768 | q8_0 | — |
| `qwen35-4b-heretic` | 99 (all 34/34) | 32768 | q8_0 | — |
| `dory-4b` | 99 (all; 72 blocks + output) | 4096 | f16 | `fa off`, 2 loops |

Layer counts are blocks plus the output layer, which is why a 32-block model reports
33 tensors and a 33-block model reports 34.

The 4B gets a large context precisely because it is small enough to afford it, and
that is where it earns its place — long document reads that the 8B and 9B cannot do
on this board. The 2-bit 9B goes further still.

### 3a. Why loads fail: the reclaim target was wrong

The single biggest fix in this repo. `reclaim-free-pages.py 2000` is **not enough**.
Loading a GGUF with `--no-mmap` leaves roughly the file's size in page cache (2.6 GB
for the 4B, 4.9 GB for the 8B). A KV `cudaMalloc` then fails for want of a contiguous
run even with `MemAvailable` reporting several GB free:

```
ggml_backend_cuda_buffer_type_alloc_buffer: allocating 544.00 MiB on device 0: cudaMalloc failed: out of memory
alloc_tensor_range: failed to allocate CUDA0 buffer of size 570425344
llama_init_from_model: failed to initialize the context: failed to allocate buffer for kv cache
```

Measured on this board: touching 5000 MB of anonymous memory drops `Cached` from
~2.6 GB to ~1.2 GB and raises `MemFree` to ~5.0 GB, after which the identical load
succeeds first try.

```bash
python3 ~/bin/reclaim-free-pages.py 5000   # not 2000
```

Consequences worth knowing:

- **The same config can fail twice and then load.** That is why every benchmark
  script here retries three times with a fresh reclaim before calling a config "does
  not fit". Single-shot measurements on this board are not trustworthy.
- **Failures cluster at the margin.** 4096 context may load first try every time
  while 6144 needs retries and 8192 never works. Report a range, not a point.
- **A background compile is enough to shift the line.** The sweeps below ran with
  the board otherwise idle; a CUDA build running 2×2 GB compilers moves the failure
  point by several GB of context. Stop the build first, or you will measure it.

## 4. Serve All Five Behind One Port

`llama-server` in router mode keeps one model resident (`--models-max 1`) and
hot-swaps on request. That is the only sane way to expose five models on 8 GB: they
cannot all be loaded at once, so the router loads whichever alias the client names
and evicts the previous one.

```bash
~/bin/start-router.sh
```

What it does, in order:

1. Kills any old `llama-server` and frees port 8080.
2. Runs the page-reclaim script so the next `cudaMalloc` gets contiguous memory.
3. Starts `llama-server --models-dir ~/models/router --models-preset ~/models/router-presets.ini --models-max 1 --port 8080`.
4. Polls `/v1/models` until the expected aliases register, then prints them.

Per-model flags live in `router-presets.ini`, not in the start script, so tuning a
model does not touch the launcher. This is the file the board actually runs, copied
2026-11-17; `scripts/router-presets.ini` in this repo is the same content.

The material change from the first version of this section is `load-mode = dio`
instead of `none`. Loading without mmap leaves the whole file in page cache, and the
nvmap/CUDA allocator on this board does not reclaim page cache to satisfy a
`cudaMalloc`, so `MemFree` collapsed and loads died with
`NvMapMemAllocInternalTagged ... error 12` even while `cudaMemGetInfo` reported GB
free. Direct I/O reads past the cache instead: `MemFree` held at 5.49 GB across a
32K-context load that previously failed. A `llama-cache-guard.timer` (fadvise
`DONTNEED`, every 2 minutes) cleans up pages left by older runs.

```ini
version = 1

; Shared across all models (Orin Nano 8GB tuning from start.sh)
; load-mode = dio (DirectIO) instead of none: with load-mode = none every load
; reads the whole GGUF into page cache and the nvmap/CUDA allocator on this board
; does NOT reclaim page cache for cudaMalloc, so MemFree collapses to ~2.9 GB and
; loads die with "NvMapMemAllocInternalTagged ... error 12 / cudaMalloc failed:
; out of memory" even while cudaMemGetInfo reports 5+ GB free. dio reads the file
; past the page cache: measured MemFree stayed at 5.49 GB across a 32k-ctx load
; that previously failed. llama-cache-guard.timer (fadvise DONTNEED, every 2 min)
; cleans up any pages left by older runs.
[*]
c = 4096
fa = on
ctk = q8_0
ctv = q8_0
np = 1
load-mode = dio
jinja = true
stop-timeout = 20

; hermes3-8b (8.03 B, 32 blocks / 33 tensors). ngl = 32 is the measured working
; point; it needs a ~4271 MiB CUDA model buffer. Lowering ngl to 24 was a
; misdiagnosis of the old page-cache OOM -- it halves generation to ~5 tok/s and
; does not fix it.
[hermes3-8b]
model = /home/snowbeach/models/router/hermes3-8b.gguf
ngl = 32

[deepseek-v4pro-9b]
model = /home/snowbeach/models/router/deepseek-v4pro-9b.gguf
ngl = 28
ub = 128

[qwen35-4b-heretic]
model = /home/snowbeach/models/router/qwen35-4b-heretic.gguf
ngl = 99
c = 32768

; Dory1.0-4B (looped latent-recurrent). The GGUF ships num_loops=4 + per_loop KV,
; which is too heavy for 8 GB; 2 loops is NVIDIA's recommended default and
; last_loop keeps the KV cache at one-loop size.
; fa off + f16 KV follow NVIDIA's launch recipe (q8_0 V cache requires flash-attn).
; reasoning off: the default thinking trace runs 2-3k tokens (~2 min at ~6 tok/s)
; before the first word. Clients can re-enable per request with
; chat_template_kwargs {"enable_thinking": true}.
[dory-4b]
model = /home/snowbeach/models/dory/NVIDIA-Nemotron-Labs-Dory1.0-4B-Q5_K_M.gguf
ngl = 99
c = 4096
fa = off
ctk = f16
ctv = f16
reasoning = off
override-kv = dory.num_loops=int:2,dory.recurrent_kv_cache_mode=str:last_loop

; DeepSeek-V4-Pro-Qwen3.5-4B-UNCENSORED-heretic (mradermacher Q4_K_M, 2.59 GiB)
; Qwen3.5-4B distill of DeepSeek-V4-Pro, heretic/abliterated. Takes after
; qwen35-4b-heretic: ngl = 99 fits in 8 GB at 32k ctx with q8_0 KV.
[deepseek-v4pro-4b-heretic]
model = /home/snowbeach/models/router/deepseek-v4pro-4b-heretic.gguf
ngl = 99
c = 32768
```

Flash attention **off** is not a preference for Dory — the looped attention path has
no fused kernel yet, and f16 KV is what the card pins.

A note on the Dory numbers below: they were measured with
`recurrent_kv_cache_mode=per_loop`, the card's default, not `last_loop` which the
board now runs. `last_loop` keeps one KV cache per recurrent layer instead of one per
loop, so it frees memory and should allow more than the 4096 measured here. That
re-measurement is on the list of things not done.

Loading without mmap (`no-mmap = true` on the stock build, `load-mode = dio` or `none`
on the fork) is deliberate. With mmap the weights are paged in from the SD card on first
touch, which on a 9B model means a slow first token and unpredictable memory
accounting on unified memory. Loading the whole file into RAM up front is faster and
makes the fit/no-fit decision deterministic. The cost is that the file also stays in
page cache — which is the whole point of §3a.

`jinja = true` turns on the model's embedded chat template. Without it, tool calls
and the reasoning-channel formatting break for the 9B.

## 5. Autocompact: keeping long chats inside the window

A 32768-token window is not enough for a working chat. The failure is blunt:

```
request (33581 tokens) exceeds the available context size (32768 tokens),
try increasing it
```

Raising `c` is not an option — §3 and §3a are the measurements that say so. The
answer is to compact instead of to grow: a small proxy sits in front of the router,
counts the prompt with the model's own tokenizer, and when the conversation crosses
a threshold it folds the older turns into a markdown checkpoint and continues in a
fresh window.

`scripts/context_guard.py` is that proxy. It listens on `127.0.0.1:8090` and
forwards to the router on `127.0.0.1:8080`:

1. counts the request with `POST /tokenize` (the model's tokenizer, not a
   character estimate);
2. if the prompt is over the trigger, folds the older messages into a summary and
   writes `~/contexts/compact-YYYYmmdd-HHMMSS.md` — YAML frontmatter
   (`tokens_before`, `tokens_after`, `messages_folded`, `messages_kept`), then
   Goal / Decisions / Artifacts / Tried and Failed / Open Questions / Next Step,
   then the full folded transcript;
3. rewrites the request as system prompt + compacted block + the last
   `keep_recent_messages` turns, and streams the reply back.

Everything else passes through untouched: same OpenAI-compatible surface, same
streaming, same tool-call fields. `GET /guard/status` reports the live config and
threshold; `X-Force-Compact: 1` forces a fold on a single request.

### The trigger is absolute, not a fraction

The guard originally triggered at `trigger_ratio` of the window — 0.80 of 32768 =
26214, which is too close to the cliff to be useful. `trigger_tokens` was added in
v1.1.0 and wins over the ratio when it is greater than zero:

```json
{ "n_ctx": 32768, "trigger_tokens": 16000, "keep_recent_messages": 6 }
```

16000 leaves room for the summarizer call, the reply, and the prefill buffers that
§Troubleshooting warns about. Config is read at startup only, so a change needs
`systemctl --user restart llama-context-guard`.

### Two things that will bite you

**The tunnel has to point at the guard, not the router.** Forwarding
`127.0.0.1:8080 -> jetson:8080` bypasses the guard completely and hands the user the
raw context error. Forward `-> jetson:8090` instead; then the URL the user already
has (`http://localhost:8080/#/`) is compacted with no client change.

**The guard must forward `Accept-Encoding`.** llama-server serves its web UI
gzip-only and answers `415 Error: gzip is not supported by this browser` to clients
that do not advertise gzip. The original `_forward()` passed only `Content-Type`,
`Authorization` and `Accept`, so every static asset and `/sw.js` 415'd behind the
guard and the browser UI broke. Response headers are relayed verbatim, so forwarding
the request header is the whole fix.

### Do not test it with a 30k payload

A 39,079-token test folded correctly (39079 -> 11193, 13 messages, 122 KB handoff)
and then put the board into a swap storm: ping still answered, sshd accepted TCP,
but the banner exchange timed out and six retries over two minutes failed. Power
cycle required. Real chats compact at 16000, well below that cliff. Verify with a
~3k history plus `X-Force-Compact: 1`:

```bash
curl -sS -m 300 http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' -H 'X-Force-Compact: 1' \
  -d '{"model":"qwen35-4b-heretic","messages":[...],"max_tokens":80}'
# 3745 -> 2711 tokens, 31.5s, handoff written, reply "The trigger threshold is 16,000 tokens."
```

## Measured performance

Every number comes from `llama-server`'s own `timings` object on this board:
`predicted_per_second` for generation, `prompt_per_second` for prefill. Generation
is 128 forced tokens (`ignore_eos`, `temperature 0`) so a model that would answer
early cannot inflate the number. Prefill is reported at two prompt lengths because
longer prefills amortize kernel launches. Three warm runs each; ranges are the
spread across runs and re-loads.

| Model | Quant | Params | Layers on GPU | Context served | Prefill tok/s (0.5K → 2.2K) | Gen tok/s | Cold load (card) | Cold swap (NVMe) | GPU model buf | KV cache | Peak RSS |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `qwen35-4b-heretic` | Q4_K_M | 4.21 B | 34/34 (ngl 99) | 32768 | 380 → 474 | 15.8 – 16.1 | 21 s | **6.6 s** | 2573 MiB | 272 MiB | 3.9 GB |
| `hermes3-8b` | Q4_K_M | 8.03 B | 32/33 (ngl 32) | 4096 | 300 → 383 | 10.0 – 10.1 | 65 s | **6.6 s** | 4271 MiB | 264 MiB | 5.4 GB |
| `deepseek-v4pro-9b` | i1 Q4_K_M | 8.95 B | 28/34 (ngl 28) | 4096 | 251 → 302 | 10.0 – 10.4 | 71 s | **9.4 s** | 4812 MiB | 68 MiB | 5.9 GB |
| `deepseek-v4pro-4b-heretic` | Q4_K_M | 4 B class | 34/34 (ngl 99) | 32768 | not measured | not measured | — | **6.0 s** | — | — | — |
| `dory-4b` (2 loops) | Q5_K_M | 4 B | ngl 99 (72 blocks + output) | 4096 | 108–127 → 206 | 6.2 – 6.7 | 48 s | **9.8 s** | 2419 MiB (+1074 host) | 208 MiB | 4.6 GB |
| `deepseek-v4pro-9b-q2` *(retired)* | i1 Q2_K | 8.95 B | 34/34 | 77824 | — → 232–240 | 9.8 | 49 s | — | 3321 MiB | 1292 MiB | 5.7 GB |

The throughput, buffer and RSS columns are the original sweep and are unchanged: they
are prefill and generation at shallow context, and nothing about the NVMe move or the
`dio` switch changes them. The `Cold load (card)` column is the pre-move measurement
and is kept for comparison. `Cold swap (NVMe)` is the end-to-end switch measured
2026-11-17 with `load-mode = dio` (load + prefill + 8 tokens). `deepseek-v4pro-4b-heretic`
has only the cold-swap number — it was added after the throughput sweep and has not
been benched.

Reading the table:

- **Generation is memory-bandwidth-bound and tracks bits moved.** 4B Q4 does
  ~16 tok/s, 8B Q4 ~10, 9B Q4 ~10, 9B Q2 ~9.8. Halving the 9B's bits gains almost
  nothing per token — the win at Q2 was the 76K context, not speed.
- **Dory is the slow one per token** (6.2–6.7 tok/s at two loops) because two loops
  execute twice the layers per token. Its payoff is fewer tokens per answer, not
  faster tokens. One loop would be roughly twice as fast for less quality — that is
  the tradeoff `dory.num_loops` exists to expose.
- **Prefill runs 2–3× generation throughput** and rises with prompt length: the 4B
  goes 380 → 474 tok/s between a 514- and a 2203-token prompt.
- **Cold load is the router's real cost, and the NVMe move paid for itself.** Swapping
  used to cost 48–71 s of waiting on the microSD; it is now 6.0–9.8 s end to end for
  every alias (see [Cold loads](#cold-loads-why-the-ggufs-moved-to-nvme)). Anything
  latency-sensitive should still pin its model rather than let `--models-max 1` evict
  it, but the penalty for switching is now ~6–10 s, not ~70.
- **KV cost per token**, measured: 17 KiB for the 4B, 66 KiB for the 8B, 17 KiB for
  the 9B (hybrid linear attention), 52 KiB for Dory at f16. That is why the 8B hits
  its wall at 8K while the 9B Q2 reached 76K.

Raw data: `results/survey.jsonl`, `results/q2-bench.json`, `results/h8b.json`,
`results/dory-bench.json`, `results/q4ngl28.log`.

## Ceilings: how far each model actually goes

Trained context is read from GGUF metadata, not from the model card. "Reliable"
means the config loaded and served on the first attempt with ~5 GB of page cache
reclaimed and the board otherwise idle; "max" means it loaded after retries. Every
probe is a real load plus a generation, never an extrapolation.

| Model | Trained ctx | Reliable ceiling | Max with retries | What limits it |
|---|---|---|---|---|
| `qwen35-4b-heretic` (ngl 99) | 262144 | **16384** | 16384 | KV (17 KiB/token) + 2573 MiB weights |
| `hermes3-8b` (ngl 32) | 131072 | **4096** | 8192 | KV (66 KiB/token) |
| `deepseek-v4pro-9b` Q4 (ngl 99) | 262144 | **4096** | 6144 | 4812 MiB model buffer leaves little for KV |
| `deepseek-v4pro-9b` Q4 (ngl 28) | 262144 | 4096 | 6144 | same — dropping layers buys nothing |
| `deepseek-v4pro-9b` Q2 (ngl 99) *(retired)* | 262144 | **77824** | 77824 | 1292 MiB KV + 3321 MiB weights |
| `dory-4b` (ngl 99, 2 loops) | 262144 | **4096** | 4096 | 2419 MiB GPU + 1074 MiB host weights |

What the sweep actually returned, so the shape is visible:

```
qwen35-4b   ngl 99: 8192 OK(3 tries)  16384 OK   18432 FAIL×3  20480 FAIL×3  24576 FAIL×3  32768 FAIL×3
hermes3-8b  ngl 32: 4096 OK            8192 OK(2) 10240 FAIL×3  12288 FAIL×3  16384 FAIL×3
9b-q4       ngl 99: 4096 OK            6144 OK(2) 8192 FAIL×3
9b-q4       ngl 28: 4096 OK            6144 OK    7168 FAIL×3   8192 FAIL×3
9b-q2       ngl 99: 8192 / 16384 / 32768 / 65536 / 73728 / 77824 OK   79872 FAIL×3  98304 FAIL×3  131072 FAIL×3
dory        ngl 99: 4096 OK            6144 FAIL×3  8192 FAIL×3
```

**This table is a floor under `load-mode = dio`, not a current maximum.** The sweep ran
with `load-mode = none`, where page-cache pressure rather than KV size was setting the
failure point. On 2026-11-17 the live preset served `qwen35-4b-heretic` at 32768 and
`deepseek-v4pro-4b-heretic` at 32768, and both loaded on the first try. Nothing above
16384 for the 4B has been re-swept under `dio`, so the real ceiling is higher and
unmeasured. Re-running `survey-ceiling.py` with `load-mode = dio` is the open task.


Two findings that contradict the earlier version of this README:

1. **The 9B offloads fully.** 34/34 tensors (33 blocks + output), `ngl 99`, 4812 MiB
   of GPU model buffer. The old "28 layers is the ceiling" was page-cache
   fragmentation (§3a), not capacity. Dropping to 28 layers buys no extra context
   (still ~6144). The served preset nonetheless runs `ngl 28`, because at 4096
   context the extra layers change nothing measurable and leaving them off keeps
   CUDA-allocator headroom for the next model's load.
2. **The 4B does not comfortably do 32K.** Its reliable ceiling is 16384. A 32K load
   succeeded once, early, on a fresher page cache — exactly the kind of single
   observation that should not become a config default.

## Verify

```bash
# all aliases present
curl -s http://127.0.0.1:8080/v1/models | python3 -m json.tool

# force a load and time the first token
curl -s http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"hermes3-8b","messages":[{"role":"user","content":"Say hi in five words."}],"max_tokens":32}'

# swap to the 9B; this evicts the 8B
curl -s http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"deepseek-v4pro-9b","messages":[{"role":"user","content":"2+2? Think it through."}],"max_tokens":128}'
```

A cold swap costs the full load time of the target model — 6.0 s to 9.8 s measured on
the NVMe, against 48–71 s when the weights were on the card — and the first request
after a swap is the slow one. Pin the model if latency matters.

To reproduce the numbers rather than just check that it answers:

```bash
python3 scripts/bench-model.py <alias> <gguf> <ngl> <ctx>   # load, buffers, RSS, pp/tg
python3 scripts/survey-ceiling.py models.json               # layer + context ceiling, with retries
python3 scripts/survey-dory.py <dory.gguf>                  # same, on the Dory fork
python3 scripts/bench-dory.py <dory.gguf> 99 4096 2         # Dory throughput at 2 loops
```

`survey-ceiling.py` takes a JSON list of `{"alias", "gguf", "extra"}` entries, retries
every probe three times, and writes `survey.jsonl`. Per-probe logs stay in
`/tmp/sv-*.log` on the board and are the receipts for every number above.

## Reaching it From a Laptop

The router only listens on the board. On the Mac, forward it and open the web UI:

```bash
~/bin/jetson-tunnel.sh
# local 127.0.0.1:8080 -> jetson 127.0.0.1:8090 (context guard) -> 127.0.0.1:8080 (router)
# then: http://localhost:8080/
```

`scripts/jetson-tunnel.sh` is that script. It probes the LAN address
(`10.0.0.13:22`) first and falls back to the tailnet IP, then execs ssh. The
fallback matters: the Tailscale path between the Mac and this board drops every few
minutes — tailscaled churns the peer endpoint between `10.0.0.181:41641` and
`[fe80::…%en0]:41641` — and a tunnel pinned to the tailnet IP dies mid-request,
which surfaces as "Server unavailable" in the UI and `ConnectionResetError` on long
completions. `tailscale down && tailscale up` on the Mac clears it temporarily; the
LAN-first script is the durable fix. launchd (`ai.hermes.jetson-llama`) runs it with
`KeepAlive`, so every restart re-picks the best path.

That is the same UI as `http://<board>:8080/`, with the model selector in the
sidebar. `?new_chat=true#/` opens straight into a fresh chat. Because the tunnel
lands on the guard, long chats compact instead of failing (§5).

## Troubleshooting

**"failed to allocate buffer" / load dies at n layers.**
Page cache is holding memory you think is free. Run
`python3 ~/bin/reclaim-free-pages.py 5000` (5000, not 2000 — see §3a), wait a few
seconds, retry. If it fails three times at the same size, that config really does
not fit: back off the context first, then `ngl`. On this board context is almost
always the cheaper thing to give up.

Measured again 2026-11-12: `hermes3-8b` at its documented `ngl = 32` (4271 MiB model
buffer) failed the KV `cudaMalloc` (263 MiB) right after a `5000` reclaim, and loaded
first try after `6000`. A 5.3 GB `deepseek-v4pro-9b` load had already put its file size
back in page cache, so the reclaim target has to cover the model you just evicted plus
the one you are loading — size it against the larger of the two, not a fixed 5000.

**Load succeeds, the first long prompt kills the server.**
Prefill allocates its own compute buffers on top of weights + KV. Leave 700–900 MiB
of headroom past the KV size, or cut `ub`.

**"failed to load model" in under a second, no other error.**
Wrong binary. Dory's GGUF needs the fork from §1b; the stock build cannot read it.

**Router answers but lists only one model.**
A single-model systemd unit is probably holding port 8080. `start-router.sh` kills
`llama-server` by name first; check `systemctl --user list-units | grep -i llama`.

**Model loads but generation is slow or the board thrashes.**
Context is too large for what is left after weights. Cut `c`, or lower `ngl` so more
of the model stays on CPU. Also check power mode: `nvpmodel -q`. At 15 W the board is
deliberately throttled; `MAXN_SUPER` is what every number here was taken at.

**Everything works, then fails after a few days uptime.**
Same fragmentation. The reclaim script exists for this; run it before each router
start, or put it in a timer.

## Status: what works, what is not done

Working, measured on the board: the CUDA build, the Dory fork build, all five models
loading and answering behind one router, per-model `ngl` ladders, the page-reclaim
step, readiness polling in `start-router.sh`, per-model prefill / generation /
context ceilings from the sweep above, and autocompact (§5) — a 33,581-token chat
that used to fail now folds at 16000 and continues.

Not done:

1. **Uptime is days, not weeks.** Everything works, then allocation fails after a
   few days. `llama-cache-guard.timer` on the board now runs fadvise `DONTNEED` every
   two minutes, which is the mitigation; it is not in this repo yet.
2. **No automated regression test.** The numbers were produced by the scripts in
   `scripts/` and are re-runnable, but nothing fails when one regresses. A
   pass/fail check at each model's ceiling is the next step.
3. **One board, one JetPack version.** R36.4.3 is what these numbers come from. A
   different L4T release may pick a different `sm` target for the CUDA build.
4. **Prefill at depth is unmeasured.** Throughput here is at shallow context. The
   interesting number — tok/s with the KV cache 90% full — is not in this README.
5. **Dory is measured at two loops only.** Loop count changes both speed and
   quality; 1, 2 and 4 loops should each get a line in the table.
6. **Presets are hand-tuned.** `router-presets.ini` is the result of trying values
   on this board, not a derived rule; another 8 GB board may need different numbers.

## Files in this repo

| Path | What |
|---|---|
| `scripts/start-router.sh` | Router launcher with page-reclaim and readiness polling |
| `scripts/router-presets.ini` | Per-model flags and shared 8 GB tuning |
| `scripts/reclaim-free-pages.py` | Forces page-cache reclaim to defragment the CUDA allocator |
| `scripts/move_models_to_nvme.sh` | Copies the served GGUFs to the NVMe, sha256-verifies, leaves symlinks (see §Storage) |
| `scripts/start-deepseek.sh` | Single-model launcher with the ngl fallback ladder |
| `scripts/bench-model.py` | Load time, buffers, RSS, prefill and generation tok/s for one model |
| `scripts/survey-ceiling.py` | Layer-count and context ceiling search, with retries |
| `scripts/survey-dory.py` | The same search against the Dory fork's `llama-server` |
| `scripts/bench-dory.py` | Dory throughput at a given loop count |
| `scripts/context_guard.py` | Autocompact proxy: folds long chats into markdown checkpoints (§5) |
| `scripts/guard-config.json` | Guard settings — `n_ctx`, `trigger_tokens`, `keep_recent_messages` |
| `scripts/jetson-tunnel.sh` | Mac-side LAN-first ssh tunnel to the guard |
| `results/` | Raw JSON from the runs behind every number in this README, plus `nvme-move.log` (the verified copy/link run) |

## Licence and Model Terms

The scripts here are yours to reuse. The models carry their own licences — check
each Hugging Face repo. The two "heretic" builds are refusal-ablated: they will
produce content that aligned models filter. That is the point of them, and it is
also your responsibility. Dory ships under the OpenMDW 1.1 licence.
