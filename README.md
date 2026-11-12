# Jetson Orin Nano Super  *Local Model Setup

How four GGUF models were added to a Jetson Orin Nano Super Developer Kit (8 GB
unified memory), served behind one `llama.cpp` router, and kept inside the memory
budget. Everything here was run on the board; numbers are measured, not estimated.

Board: `snowbeach` — Jetson Orin Nano Super Developer Kit, aarch64, Ubuntu 22.04,
JetPack R36.4.3, CUDA 12.6, 8 GB unified LPDDR5, power mode `MAXN_SUPER`.

Measured on llama.cpp build `10291 (7e30f3adb)` for the Qwen and DeepSeek builds,
and on the Dory fork build `1 (98a74c9)` for Dory — see
[Measured performance](#measured-performance).

## Storage

The kit boots from the 512 GB microSD card; the NVMe drive is secondary storage.

| Device | Size | Mount | Used | Free |
|---|---|---|---|---|
| `/dev/mmcblk0p1` (512 GB microSD) | 475.3 G | `/` | 249 G | 200 G |
| `/dev/nvme0n1p1` (NVMe SSD) | 238.5 G | `/mnt/nvme` | 119 G | 104 G |

The NVMe in this build is 256 GB class, not 8 GB — the "8 GB" in the kit name is
the unified RAM. Keep that straight: 8 GB is the hard ceiling on what can be
resident, and storage is not the constraint.

Models live under `/home/snowbeach/models` (microSD, root filesystem). They were
deliberately *not* put on the NVMe so a card swap does not lose them, and because
root has more headroom.

## The Models

| Alias | File | Quant | Size | Source |
|---|---|---|---|---|
| `hermes3-8b` | `hermes3-8b.gguf` | Q4_K_M | 4.92 GB | [bartowski/Hermes-3-Llama-3.1-8B-lorablated-GGUF](https://huggingface.co/bartowski/Hermes-3-Llama-3.1-8B-lorablated-GGUF) |
| `deepseek-v4pro-9b` | `deepseek-v4pro-9b.gguf` | i1 Q4_K_M | 5.63 GB | [mradermacher/DeepSeek-V4-Pro-Qwen3.5-9B-UNCENSORED-heretic-i1-GGUF](https://huggingface.co/mradermacher/DeepSeek-V4-Pro-Qwen3.5-9B-UNCENSORED-heretic-i1-GGUF) |
| `qwen35-4b-heretic` | `qwen35-4b-heretic.gguf` | Q4_K_M | 2.71 GB | [mradermacher/Qwen3.5-4B-heretic-GGUF](https://huggingface.co/mradermacher/Qwen3.5-4B-heretic-GGUF) |
| `deepseek-v4pro-9b-q2` | `…heretic.i1-Q2_K.gguf` | i1 Q2_K | 3.83 GB | same repo as the 9B above |
| `dory` | `NVIDIA-Nemotron-Labs-Dory1.0-4B-Q5_K_M.gguf` | Q5_K_M | 3.67 GB | [nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-GGUF](https://huggingface.co/nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-GGUF) |

What Each One Is:

- **Hermes3-8b** — Hermes 3 Llama 3.1 8B, loRA-ablated, then imatrix-quantized by
  bartowski. The most reliable general chat model here, and the one with the least
  room to spare. `ngl 99` (4403 MiB model buffer) is right at the edge: it loaded
  for the throughput runs and failed 3/3 in the ceiling sweep, while `ngl 32`
  (4271 MiB, still 33 of 34 layers) loaded first try every time. The 132 MiB
  difference is the output layer, and on an 8 GB board that is the margin.
- **Deepseek-v4pro-9b (Q4)** — DeepSeek-V4-Pro distilled into Qwen3.5-9B, then
  "heretic" refusal-ablated (ARA). 8.95 B params, 32 transformer layers, hybrid
  linear+full attention, native 262K context. The strongest reasoner here. Earlier
  notes claimed it could only offload 28 of 32 layers; with the page-reclaim fix in
  §3a it offloads all 34/34, and layers are not what limits it — context is.
- **Qwen35-4b-heretic** — Qwen3.5-4B, language tower ablated (vision encoder left
  alone). 4 B params, 262K trained context. The cheap fast one, and the best of the
  three for long inputs that fit inside 16K.
- **Deepseek-v4pro-9b-q2** — the 2-bit fallback, 3.83 GB. Trades quality for
  headroom: at Q2 the same 9B serves a **76K** context on this board, which nothing
  else here approaches.
- **Dory** — NVIDIA Nemotron Labs Dory1.0-4B, a *looped* (latent-recurrent)
  normalized-GPT: the shared recurrent block is applied N times at inference. Needs
  its own llama.cpp fork (§1b) and `dory.num_loops` overrides. Two loops is the
  setting measured here. The most token-efficient per parameter and the slowest per
  token; 113 layers in the file, 99 of them offloaded.

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
so all four models — Dory included — are served by one process on one port. The fork
also dropped `--no-mmap`: the replacement is `load-mode = none` in the preset file.

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
| `hermes3-8b` | 32 of 34 | 4096 | q8_0 | — |
| `deepseek-v4pro-9b` | 99 (all 34/34) | 4096 | q8_0 | `ub 128` |
| `deepseek-v4pro-9b-q2` | 99 (all) | 32768–77824 | q8_0 | `ub 128` |
| `qwen35-4b-heretic` | 99 (all) | 16384 | q8_0 | — |
| `dory` | 99 of 113 | 4096 | f16 | `fa off`, 2 loops |

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

## 4. Serve All Four Behind One Port

`llama-server` in router mode keeps one model resident (`--models-max 1`) and
hot-swaps on request. That is the only sane way to expose four models on 8 GB: they
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
model does not touch the launcher:

```ini
version = 1

; fork rev 98a74c9 removed --no-mmap; the equivalent is load-mode = none
[*]
c = 4096
fa = on
ctk = q8_0
ctv = q8_0
np = 1
load-mode = none
jinja = true
stop-timeout = 20

[hermes3-8b]
model = /home/snowbeach/models/router/hermes3-8b.gguf
ngl = 32          ; 99 also works but is 132 MiB from not fitting

[deepseek-v4pro-9b]
model = /home/snowbeach/models/router/deepseek-v4pro-9b.gguf
ngl = 99          ; all 34/34 layers fit; context is the constraint, not layers
ub = 128

[qwen35-4b-heretic]
model = /home/snowbeach/models/router/qwen35-4b-heretic.gguf
ngl = 99
c = 16384

[deepseek-v4pro-9b-q2]
model = /home/snowbeach/models/router/deepseek-v4pro-9b-q2.gguf
ngl = 99
c = 65536
ub = 128

; Dory needs per-model flags the shared block cannot express. The GGUF ships
; num_loops=4 + per_loop KV, too heavy for 8 GB; 2 loops is NVIDIA's recommended
; default and last_loop keeps the KV cache at one-loop size. fa off + f16 KV
; follow NVIDIA's launch recipe (a q8_0 V cache requires flash attention).
; reasoning is off by default: the thinking trace runs 2-3k tokens — roughly two
; minutes at ~6 tok/s — before the first word. Clients can re-enable it per
; request with chat_template_kwargs {"enable_thinking": true}.
[dory-4b]
model = /home/snowbeach/models/dory/NVIDIA-Nemotron-Labs-Dory1.0-4B-Q5_K_M.gguf
ngl = 99
c = 4096
fa = off
ctk = f16
ctv = f16
reasoning = off
override-kv = dory.num_loops=int:2,dory.recurrent_kv_cache_mode=str:last_loop
```

Flash attention **off** is not a preference for Dory — the looped attention path has
no fused kernel yet, and f16 KV is what the card pins.

A note on the Dory numbers below: they were measured with
`recurrent_kv_cache_mode=per_loop`, the card's default, not `last_loop` which the
board now runs. `last_loop` keeps one KV cache per recurrent layer instead of one per
loop, so it frees memory and should allow more than the 4096 measured here. That
re-measurement is on the list of things not done.

Loading without mmap (`no-mmap = true` on the stock build, `load-mode = none` on the
fork) is deliberate. With mmap the weights are paged in from the SD card on first
touch, which on a 9B model means a slow first token and unpredictable memory
accounting on unified memory. Loading the whole file into RAM up front is faster and
makes the fit/no-fit decision deterministic. The cost is that the file also stays in
page cache — which is the whole point of §3a.

`jinja = true` turns on the model's embedded chat template. Without it, tool calls
and the reasoning-channel formatting break for the 9B.

## Measured performance

Every number comes from `llama-server`'s own `timings` object on this board:
`predicted_per_second` for generation, `prompt_per_second` for prefill. Generation
is 128 forced tokens (`ignore_eos`, `temperature 0`) so a model that would answer
early cannot inflate the number. Prefill is reported at two prompt lengths because
longer prefills amortize kernel launches. Three warm runs each; ranges are the
spread across runs and re-loads.

| Model | Quant | Params | Layers on GPU | Context served | Prefill tok/s (0.5K → 2.2K) | Gen tok/s | Cold load | GPU model buf | KV cache | Peak RSS |
|---|---|---|---|---|---|---|---|---|---|---|
| `qwen35-4b-heretic` | Q4_K_M | 4.21 B | 33/33 | 16384 | 380 → 474 | 15.8 – 16.1 | 21 s | 2573 MiB | 272 MiB | 3.9 GB |
| `hermes3-8b` | Q4_K_M | 8.03 B | 33/34 (ngl 32) | 4096 | 300 → 383 | 10.0 – 10.1 | 65 s | 4271 MiB | 264 MiB | 5.4 GB |
| `deepseek-v4pro-9b` | i1 Q4_K_M | 8.95 B | 34/34 (ngl 99) | 4096 | 251 → 302 | 10.0 – 10.4 | 71 s | 4812 MiB | 68 MiB | 5.9 GB |
| `deepseek-v4pro-9b-q2` | i1 Q2_K | 8.95 B | 34/34 | 77824 | — → 232–240 | 9.8 | 49 s | 3321 MiB | 1292 MiB | 5.7 GB |
| `dory` (2 loops) | Q5_K_M | 4 B | 99/113 | 4096 | 108–127 → 206 | 6.2 – 6.7 | 48 s | 2419 MiB (+1074 host) | 208 MiB | 4.6 GB |

Reading the table:

- **Generation is memory-bandwidth-bound and tracks bits moved.** 4B Q4 does
  ~16 tok/s, 8B Q4 ~10, 9B Q4 ~10, 9B Q2 ~9.8. Halving the 9B's bits gains almost
  nothing per token — the win at Q2 is the 76K context, not speed.
- **Dory is the slow one per token** (6.2–6.7 tok/s at two loops) because two loops
  execute twice the layers per token. Its payoff is fewer tokens per answer, not
  faster tokens. One loop would be roughly twice as fast for less quality — that is
  the tradeoff `dory.num_loops` exists to expose.
- **Prefill runs 2–3× generation throughput** and rises with prompt length: the 4B
  goes 380 → 474 tok/s between a 514- and a 2203-token prompt.
- **Cold load is the router's real cost.** Swapping from the 4B to the 9B costs 71 s
  of waiting, so anything latency-sensitive should pin its model rather than let
  `--models-max 1` evict it.
- **KV cost per token**, measured: 17 KiB for the 4B, 66 KiB for the 8B, 17 KiB for
  the 9B (hybrid linear attention), 52 KiB for Dory at f16. That is why the 8B hits
  its wall at 8K while the 9B Q2 reaches 76K.

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
| `deepseek-v4pro-9b` Q2 (ngl 99) | 262144 | **77824** | 77824 | 1292 MiB KV + 3321 MiB weights |
| `dory` (ngl 99, 2 loops) | 262144 | **4096** | 4096 | 2419 MiB GPU + 1074 MiB host weights |

What the sweep actually returned, so the shape is visible:

```
qwen35-4b   ngl 99: 8192 OK(3 tries)  16384 OK   18432 FAIL×3  20480 FAIL×3  24576 FAIL×3  32768 FAIL×3
hermes3-8b  ngl 32: 4096 OK            8192 OK(2) 10240 FAIL×3  12288 FAIL×3  16384 FAIL×3
9b-q4       ngl 99: 4096 OK            6144 OK(2) 8192 FAIL×3
9b-q4       ngl 28: 4096 OK            6144 OK    7168 FAIL×3   8192 FAIL×3
9b-q2       ngl 99: 8192 / 16384 / 32768 / 65536 / 73728 / 77824 OK   79872 FAIL×3  98304 FAIL×3  131072 FAIL×3
dory        ngl 99: 4096 OK            6144 FAIL×3  8192 FAIL×3
```

Two findings that contradict the earlier version of this README:

1. **The 9B offloads fully.** 34/34 layers, `ngl 99`, 4812 MiB of GPU model buffer.
   The old "28 layers is the ceiling" was page-cache fragmentation (§3a), not
   capacity. Dropping to 28 layers buys no extra context (still ~6144), so there is
   no reason to run it that way.
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

A cold swap costs the full load time of the target model — 21 s to 71 s, measured —
and the first request after a swap is the slow one. Pin the model if latency matters.

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
ssh -N -T -o ExitOnForwardFailure=yes -o ServerAliveInterval=15 jetson \
  -L 127.0.0.1:8090:127.0.0.1:8080
# then: http://localhost:8090/
```

That is the same UI as `http://<board>:8080/`, with the model selector in the
sidebar. `?new_chat=true#/` opens straight into a fresh chat.

## Troubleshooting

**"failed to allocate buffer" / load dies at n layers.**
Page cache is holding memory you think is free. Run
`python3 ~/bin/reclaim-free-pages.py 5000` (5000, not 2000 — see §3a), wait a few
seconds, retry. If it fails three times at the same size, that config really does
not fit: back off the context first, then `ngl`. On this board context is almost
always the cheaper thing to give up.

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

Working, measured on the board: the CUDA build, the Dory fork build, all four models
loading and answering behind one router, per-model `ngl` ladders, the page-reclaim
step, readiness polling in `start-router.sh`, and per-model prefill / generation /
context ceilings from the sweep above.

Not done:

1. **Uptime is days, not weeks.** Everything works, then allocation fails after a
   few days. The reclaim script exists but is not wired to a timer here.
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
| `scripts/start-deepseek.sh` | Single-model launcher with the ngl fallback ladder |
| `scripts/bench-model.py` | Load time, buffers, RSS, prefill and generation tok/s for one model |
| `scripts/survey-ceiling.py` | Layer-count and context ceiling search, with retries |
| `scripts/survey-dory.py` | The same search against the Dory fork's `llama-server` |
| `scripts/bench-dory.py` | Dory throughput at a given loop count |
| `results/` | Raw JSON from the runs behind every number in this README |

## Licence and Model Terms

The scripts here are yours to reuse. The models carry their own licences — check
each Hugging Face repo. The two "heretic" builds are refusal-ablated: they will
produce content that aligned models filter. That is the point of them, and it is
also your responsibility. Dory ships under the OpenMDW 1.1 licence.
