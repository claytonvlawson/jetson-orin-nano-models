# Jetson Orin Nano Super  *Local Model Setup

How three GGUF models were added to a Jetson Orin Nano Super Developer Kit (8 GB
unified memory), served behind one `llama.cpp` router, and kept inside the memory
budget. Everything here was run on the board; numbers are measured, not estimated.

Board: `snowbeach` — Jetson Orin Nano Super Developer Kit, aarch64, Ubuntu 22.04,
JetPack R36.4.3, CUDA 12.6, 8 GB unified LPDDR5.

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

## The Three Models

| Alias | File | Quant | Size | Source |
|---|---|---|---|---|
| `hermes3-8b` | `hermes3-8b.gguf` | Q4_K_M | 4.92 GB | [bartowski/Hermes-3-Llama-3.1-8B-lorablated-GGUF](https://huggingface.co/bartowski/Hermes-3-Llama-3.1-8B-lorablated-GGUF) |
| `deepseek-v4pro-9b` | `deepseek-v4pro-9b.gguf` | i1 Q4_K_M | 5.63 GB | [mradermacher/DeepSeek-V4-Pro-Qwen3.5-9B-UNCENSORED-heretic-i1-GGUF](https://huggingface.co/mradermacher/DeepSeek-V4-Pro-Qwen3.5-9B-UNCENSORED-heretic-i1-GGUF) |
| `qwen35-4b-heretic` | `qwen35-4b-heretic.gguf` | Q4_K_M | 2.71 GB | [mradermacher/Qwen3.5-4B-heretic-GGUF](https://huggingface.co/mradermacher/Qwen3.5-4B-heretic-GGUF) |

What Each One Is:

- **Hermes3-8b** — Hermes 3 Llama 3.1 8B, loRA-ablated, then imatrix-quantized by
  bartowski. The most reliable general chat model of the three. Loads to 99 GPU
  layers without complaint, so it gets the full card.
- **Deepseek-v4pro-9b** — DeepSeek-V4-Pro distilled into Qwen3.5-9B, then
  "heretic" refusal-ablated (ARA). 8.95 B params, 32 transformer layers, hybrid
  linear+full attention, native 262K context. The strongest reasoner here, and the
  only one that cannot fit entirely in 8 GB at Q4_K_M.
- **Qwen35-4b-heretic** — Qwen3.5-4B, language tower ablated (vision encoder left
  alone). 4 B params, 32K context fits comfortably, so it gets `ngl 99` and a much
  larger context window than the other two.

An additional Q2_K build of the 9B (`DeepSeek-V4-Pro-Qwen3.5-9B-UNCENSORED-heretic.i1-Q2_K.gguf`,
3.83 GB) is kept next to the Q4 for the fallback path — it loads when Q4 at reduced
layer counts still runs out of memory.

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
```

Then stage flat copies in the router directory, because the router keys models by
the filename in its `--models-dir`:

```bash
mkdir -p ~/models/router
cp ~/models/Hermes-3-Llama-3.1-8B-lorablated-GGUF/Hermes-3-Llama-3.1-8B-lorablated-Q4_K_M.gguf ~/models/router/hermes3-8b.gguf
cp ~/models/deepseek-v4pro-9b/DeepSeek-V4-Pro-Qwen3.5-9B-UNCENSORED-heretic.i1-Q4_K_M.gguf ~/models/router/deepseek-v4pro-9b.gguf
cp ~/models/qwen35-4b-heretic/Qwen3.5-4B-heretic.Q4_K_M.gguf ~/models/router/qwen35-4b-heretic.gguf
```

The 9B quant name in this repo follows mradermacher's `.i1-Q4_K_M` suffix, while the
4B and 8B use plain `-Q4_K_M`. Check the actual filename in the repo's file list
before downloading; renaming after download is fine.

## 3. Tune for 8 GB

Three constraints drive every flag:

1. **8 GB is shared** between CPU and GPU. A 5.6 GB model plus a KV cache plus
   CUDA context overhead does not leave room for a 9B at 99 layers.
2. **Long-running boards fragment the CUDA allocator.** A multi-GB `cudaMalloc`
   fails on a board that has been up for days even when `free` shows room. That is
   what `reclaim-free-pages.py` is for: touch anonymous pages to force page-cache
   reclaim, then release them, so a contiguous allocation can succeed.
3. **KV cache is the second-largest allocation.** Quantizing it to `q8_0` roughly
   halves it versus f16 and is the difference between fitting and not.

### What Fit, Per Model

| Alias | GPU layers | Context | KV cache | Extra |
|---|---|---|---|---|
| `hermes3-8b` | 99 (all) | 4096 | q8_0 | — |
| `deepseek-v4pro-9b` | 28 | 4096 | q8_0 | `ub 128` batch |
| `qwen35-4b-heretic` | 99 (all) | 32768 | q8_0 | — |

The 9B was the only one that needed a search. Layer counts were tried in order —
28, 26, 24, 20, then CPU-only — and the first value that loaded cleanly was kept.
`attempt-ngl24.log`, `attempt-ngl26.log`, and `attempt-ngl28.log` in
`~/models/deepseek-v4pro-9b/` are the receipts. 28 is the ceiling: at 30 the
allocation fails.

The 4B gets a 32K context precisely because it is small enough to afford it, and
that is where it earns its place — long document reads that the 8B and 9B cannot do.

## 4. Serve All Three Behind One Port

`llama-server` in router mode keeps one model resident (`--models-max 1`) and
hot-swaps on request. That is the only sane way to expose three models on 8 GB:
they cannot all be loaded at once, so the router loads whichever alias the client
names and evicts the previous one.

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

[*]
c = 4096
fa = on
ctk = q8_0
ctv = q8_0
np = 1
no-mmap = true
jinja = true
stop-timeout = 20

[hermes3-8b]
model = /home/snowbeach/models/router/hermes3-8b.gguf
ngl = 99

[deepseek-v4pro-9b]
model = /home/snowbeach/models/router/deepseek-v4pro-9b.gguf
ngl = 28
ub = 128

[qwen35-4b-heretic]
model = /home/snowbeach/models/router/qwen35-4b-heretic.gguf
ngl = 99
c = 32768
```

`no-mmap = true` is deliberate. With mmap the weights are paged in from the SD card
on first touch, which on a 9B model means a slow first token and unpredictable
memory accounting on unified memory. Loading the whole file into RAM up front is
faster and makes the fit/no-fit decision deterministic.

`jinja = true` turns on the model's embedded chat template. Without it, tool calls
and the reasoning-channel formatting break for the 9B.

## 5. Verify

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

A cold swap costs the full load time of the target model; the first request after a
swap is the slow one. Anything latency-sensitive should pin its model and not let
the router evict it.

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
The allocator is fragmented. Run `python3 ~/bin/reclaim-free-pages.py 2000`, wait a
few seconds, retry. If it still fails, drop `ngl` by 2 (28 → 26 → 24 → 20) or fall
back to the Q2_K build. The 9B's `start.sh` does this ladder automatically.

**Router answers but lists only one model.**
A single-model systemd unit is probably holding port 8080. `start-router.sh` kills
`llama-server` by name first; check `systemctl --user list-units | grep -i llama`.

**Model loads but generation is slow or the board thrashes.**
Context is too large for what is left after weights. Cut `c`, or set `ngl` lower so
more of the model stays on CPU. Also check power mode: `nvpmodel -q`. At 15 W the
board is deliberately throttled; Super mode gives more headroom at higher draw.

**Everything works, then fails after a few days uptime.**
Same fragmentation. The reclaim script exists for this; run it before each router
start, or put it in a timer.

## Status: what works, what is not done

Working, measured on the board: the CUDA build, all three models loading and
answering behind one router, per-model `ngl` ladders, the page-reclaim step, and
the readiness polling in `start-router.sh`. This is the setup that has been
serving this board since November 2026.

Not done:

1. **Uptime is days, not weeks.** The honest finding in Troubleshooting:
   everything works, then allocation fails after a few days. The reclaim script
   exists but is not wired to a timer here.
2. **No automated test.** The verification is the transcript in the README plus
   `start-router.sh`'s readiness poll. A script that loads each model and asks
   one question would turn the numbers into a check; it is not written.
3. **One board, one JetPack version.** R36.4.3 is what these numbers come from.
   A different L4T release may pick a different `sm` target for the CUDA build.
4. **No token/serving benchmark.** Throughput is not measured, only memory fit.
5. **Presets are hand-tuned.** `router-presets.ini` is the result of trying
   values on this board, not a derived rule; another 8 GB board may need
   different numbers.

## Files in this repo

| Path | What |
|---|---|
| `scripts/start-router.sh` | Router launcher with page-reclaim and readiness polling |
| `scripts/router-presets.ini` | Per-model flags and shared 8 GB tuning |
| `scripts/reclaim-free-pages.py` | Forces page-cache reclaim to defragment the CUDA allocator |
| `scripts/start-deepseek.sh` | Single-model launcher with the ngl fallback ladder |

## Licence and Model Terms

The scripts here are yours to reuse. The models carry their own licences — check
each Hugging Face repo. The two "heretic" builds are refusal-ablated: they will
produce content that aligned models filter. That is the point of them, and it is
also your responsibility.
