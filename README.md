# Dual RTX 2080 Ti 22GB — vLLM Runtime Recipe

How I run a Qwen 3.8 27B class model on **two modded RTX 2080 Ti (22GB each)** with
NVLink, using a custom vLLM build targeting `sm_75`.

This repo documents the **runtime configuration**: the exact launch flag set, the
profile files behind it, and the operational glue (GPU clock/power governor,
benchmark harnesses). It is meant as a reference for people trying to get a
27B-class model running on Turing-era hardware.

> **No model weights are included.** No quantized blobs, no merged checkpoints,
> no HF tokens. Bring your own weights — see [Models](#models).

---

## Hardware

| Item | Value |
| --- | --- |
| GPU | 2 × NVIDIA GeForce RTX 2080 Ti, **22528 MiB (22GB) each** — memory-modded |
| Interconnect | NVLink 2-way (`NV2`), 25.781 GB/s per link |
| Topology | Single NUMA node, both GPUs on CPU affinity `0-7` |
| CPU | Intel Core i7-7740X @ 4.30GHz (4C/8T) |
| RAM | 15 GB total |
| OS | Ubuntu 24.04.4 LTS, kernel 7.0.0-31-generic |
| Driver | `nvidia-driver-595-open` 595.84, CUDA 13.2 runtime capability |

Key hardware notes:

- **22GB per card is the whole game.** A stock 2080 Ti has 11GB. The memory mod
  is what makes a 27B model fit across two cards at a useful context length.
- **NVLink matters more than the extra PCIe bandwidth.** Tensor-parallel size 2
  does an all-reduce every layer; without NVLink this becomes the bottleneck on
  generative decode.
- **The i7-7740X has only 4 physical cores.** That is a real constraint for
  tokenizer/detokenizer throughput at long context, and it shapes the
  `--max-num-batched-tokens` and scheduler choices below.
- **15GB system RAM** means the model must not be loaded/reshaped host-side in
  a memory-hungry way; keep host-side work minimal.

## Stack versions

| Component | Version |
| --- | --- |
| vLLM | `0.2.1rc2` (custom `sm_75` build) |
| Base vLLM | `0.27.1` |
| PyTorch | `2.13.0+cu130` |
| CUDA | `13.0` (torch build), driver 595.84 |
| transformers | `5.15.1` |
| Python | `3.12.3` |

## Models

The configuration in this repo was developed against, and the served name
encodes:

- **Weights:** `orcarouter-Qwen3.8-27B-Uncensored-NVFP4` (27B class, NVFP4 /
  `compressed-tensors` quantization)

The model identifier is stated for reproducibility only. **The weights are not
redistributed here** — download them from their original upstream source and
adjust the `--model` path to your own checkout.

---

## The launch command

This is the live invocation the running server uses:

```bash
vllm serve /path/to/your/orcarouter-Qwen3.8-27B-Uncensored-NVFP4 \
  --host 0.0.0.0 \
  --port 8000 \
  --served-model-name qwen38-27b-orcarouter-nvfp4-fp8kv-230K-mtp5-text-only-cu130 \
  --dtype half \
  --tensor-parallel-size 2 \
  --pipeline-parallel-size 1 \
  --generation-config vllm \
  --max-model-len 230000 \
  --enable-chunked-prefill \
  --max-num-seqs 1 \
  --max-num-batched-tokens 2048 \
  --quantization compressed-tensors \
  --kv-cache-dtype fp8 \
  --kv-cache-memory-bytes 5115441742 \
  --mamba-cache-mode align \
  --enable-prefix-caching \
  --enable-prompt-tokens-details \
  --language-model-only \
  --skip-mm-profiling \
  --reasoning-parser qwen3 \
  --tool-call-parser qwen3_xml \
  --enable-auto-tool-choice \
  --additional-config '{"gdn_prefill_backend":"flashqla_legacy"}' \
  --speculative-config '{"method":"mtp","num_speculative_tokens":5}' \
  --compilation-config '{"cudagraph_mode":"PIECEWISE","cudagraph_capture_sizes":[6,2048],"max_cudagraph_capture_size":2048}'
```

An equivalent file is in [`launch/serve.sh`](launch/serve.sh).

### Why each flag is there

| Flag | Reasoning |
| --- | --- |
| `--tensor-parallel-size 2` | Splits the model across both cards. NVLink makes the per-layer all-reduce affordable. |
| `--max-model-len 230000` | 230K context. This is the tuned maximum that still leaves room for KV cache + activations within 2×22GB. |
| `--kv-cache-dtype fp8` | Halves KV cache footprint vs `fp16`, which is what makes the long context affordable on this hardware. |
| `--kv-cache-memory-bytes 5115441742` | **Explicit** KV cache budget instead of relying on `--gpu-memory-utilization` heuristics. Pinning this makes memory behaviour reproducible across restarts and avoids over-reserving. |
| `--max-num-seqs 1` | Single concurrent sequence. With 22GB/card and a 230K context the memory budget does not allow a useful batch; prioritising one long sequence is the deliberate tradeoff. |
| `--max-num-batched-tokens 2048` | Chunked prefill budget, sized to the cudagraph capture and to the modest 4-core CPU. |
| `--enable-chunked-prefill` | Keeps long prefills from blocking decode and bounds activation memory spikes. |
| `--quantization compressed-tensors` | Matches the NVFP4 checkpoint's quantization scheme. |
| `--mamba-cache-mode align` | Required by the hybrid Mamba/GDN attention architecture in this model family. |
| `--speculative-config … "mtp" … 5` | Multi-token prediction with 5 speculative tokens. This is the single biggest decode win and the main reason the config is tuned around MTP. |
| `--additional-config '{"gdn_prefill_backend":"flashqla_legacy"}'` | Selects the legacy FlashQLA GDN prefill backend, which is what works on `sm_75` after the patch work. |
| `--compilation-config … PIECEWISE` | `PIECEWISE` cudagraph mode with capture sizes `[6, 2048]`. The capture list **must include the chunked-prefill budget** (2048) as well as the decode shape (6 = `MTP_K + 1`) — a decode-only list leaves every prefill eager and adds avoidable overhead. |
| `--language-model-only` + `--skip-mm-profiling` | Text-only serving; skips multimodal profiling that would otherwise cost memory and startup time for unused vision towers. |
| `--enable-prefix-caching` | Reuses KV across shared prefixes. |
| `--dtype half` | Compute dtype; weights remain quantized. |

The `--served-model-name` deliberately encodes the whole configuration
(`…nvfp4-fp8kv-230K-mtp5-text-only-cu130`) so a client can tell at a glance which
profile answered the request.

---

## Profiles

[`profiles/`](profiles/) holds the profile files that drive the launcher. Each
profile is a plain `KEY=VALUE` env file; the launcher translates it into CLI
flags. Keeping them as files rather than shell history is what makes the tuning
reproducible and diffable.

| Profile | Context | MTP | KV dtype | Quant | Notes |
| --- | --- | --- | --- | --- | --- |
| [`orcarouter-fp8kv-230K-mtp5-text-only.env`](profiles/orcarouter-fp8kv-230K-mtp5-text-only.env) | 230K | 5 | fp8 | compressed-tensors | **Currently served config.** Text-only. |
| [`orcarouter-fp8kv-250K-mtp5-text-only.env`](profiles/orcarouter-fp8kv-250K-mtp5-text-only.env) | 250K | 5 | fp8 | compressed-tensors | Longer context, lower `gpu_util`. |
| [`orcarouter-fp8kv-200K-mtp5-text-only.env`](profiles/orcarouter-fp8kv-200K-mtp5-text-only.env) | 200K | 5 | fp8 | compressed-tensors | More headroom variant. |
| [`fp8kv-240K-mtp3-text-image.env`](profiles/fp8kv-240K-mtp3-text-image.env) | 240K | 3 | fp8 | compressed-tensors | **Text + image**; multimodal enabled. |
| [`fp8kv-240K-nomtp-text-only.env`](profiles/fp8kv-240K-nomtp-text-only.env) | 240K | 0 | fp8 | compressed-tensors | MTP disabled — the control/baseline case. |
| [`modelopt-w4a16-fp8kv-128K-mtp3-text-only.env`](profiles/modelopt-w4a16-fp8kv-128K-mtp3-text-only.env) | 128K | 3 | fp8 | `modelopt_mixed` (W4A16) | Alternative quantization path. |

Notable per-profile environment variables beyond the flags:

- `VLLM_QWOPUS_MTP_BF16_DRAFT=1` — keep the MTP draft head in BF16 for stability.
- `VLLM_ALLOW_MAMBA_SPEC_FULL_CUDAGRAPH=0` — deliberately disabled; the full
  cudagraph path for Mamba speculation was not stable here.
- `VLLM_TURBOQUANT_CONTINUATION_PREFIX_COMBINE=auto` with
  `…_MIN_TOKENS=20480` and `VLLM_TURBOQUANT_DECODE_BLOCK_KV=2` (230K profile) —
  tuning for continuation-style long-prefix workloads.
- `PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.8` — reduces allocator
  fragmentation over long uptimes.

### Mapping the profile to the flag set

The launcher logic is small. The essentials:

```bash
# context + KV budget
VLLM_ARGS+=(--max-model-len "$MAX_MODEL_LEN")
VLLM_ARGS+=(--kv-cache-dtype "$KV_CACHE_DTYPE")
VLLM_ARGS+=(--kv-cache-memory-bytes "$KV_CACHE_MEMORY_BYTES")

# speculative decoding
if (( MTP_K > 0 )); then
  VLLM_ARGS+=(--speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":${MTP_K}}")
fi

# capture BOTH the decode shape (MTP_K+1) and the prefill budget
capture=$((MTP_K + 1))
prefill_capture=${MAX_BATCHED_TOKENS:-2048}
(( prefill_capture < capture )) && prefill_capture=$capture
# -> cudagraph_capture_sizes: [capture, prefill_capture]  == [6, 2048]
```

---

## GPU clock & power governor

[`launch/vllm-clock-governor.sh`](launch/vllm-clock-governor.sh) is a small
daemon that ties GPU state to inference activity:

- Applies a **200W power limit** to both cards at startup (the cards allow up to
  280W; capping keeps the modded cards thermally sane and the fans quiet).
- **Locks graphics clocks at 1800 MHz on GPU0 only while requests are active**,
  by polling `vllm:num_requests_running` from the metrics endpoint.
- Releases the lock back to dynamic clocks after a 120s idle grace period.

The intent is to keep clocks pinned (and therefore latency predictable) during
generation, while letting the card idle normally otherwise. It needs

```
<youruser> ALL=(root) NOPASSWD: /usr/bin/nvidia-smi
```

(or equivalent) since it shells out to `sudo -n nvidia-smi`. Adjust
`POWER_LIMIT_WATTS`, `LOCK_RANGE`, and `GPU_INDEX` to taste — note the script as
written locks **GPU0 only**.

---

## Benchmarks

[`benchmarks/`](benchmarks/) contains the harnesses used during tuning. They are
included as **methodology**, not as published results — run them on your own
hardware. Paths are sanitized to `<HOME>`.

| Script | What it measures |
| --- | --- |
| [`p6_bench.sh`](benchmarks/p6_bench.sh) | Streaming run over ~4K-token prompts; reports **prefill tok/s**, **decode tok/s**, and **TTFT** by splitting the stream at first token. |
| [`p7_bench.py`](benchmarks/p7_bench.py) | Sequential, greedy, fixed distinct prompts with `ignore_eos`; per-prompt and aggregate tok/s including prefill. Appends to a summary file for A/B comparison. |
| [`p12_bench.py`](benchmarks/p12_bench.py) | Non-streaming same-domain bench. Isolates decode by differencing two runs (short vs. long generation), cancelling out prefill. |
| [`bench_p4.py`](benchmarks/bench_p4.py) / [`bench_p4b.py`](benchmarks/bench_p4b.py) | Early short-prompt throughput checks. |

### Methodology notes

- **Separate prefill from decode.** Aggregate "tok/s" hides which side you
  improved. `p6_bench.sh` splits at TTFT; `p12_bench.py` differences two runs.
- **Greedy + `ignore_eos`** for comparable token counts across configurations.
- **Fixed distinct prompts.** Reusing one prompt lets prefix caching flatter the
  numbers; the p7 prompt set is deliberately disjoint to avoid cache hits.
- **Single-request measurement.** With `--max-num-seqs 1` these are
  latency-oriented numbers, not aggregate server throughput. Do not compare them
  against batched serving benchmarks.

Example harness call:

```bash
python3 benchmarks/p7_bench.py <label>
```

---

## Repository layout

```
.
├── README.md
├── launch/
│   ├── serve.sh                  # the launch command as a script
│   └── vllm-clock-governor.sh    # GPU clock/power governor daemon
├── profiles/                     # KEY=VALUE profile files
└── benchmarks/                   # measurement harnesses + methodology
```

## What is *not* in this repo

Deliberately excluded:

- **Model weights** and any quantized artifacts.
- **HF tokens, API keys, SSH keys, passwords** — none are present. Paths and
  hostnames are sanitized to `<HOME>` / generic values.
- **Performance headline numbers** — I would rather publish measured results
  from a controlled run than quote unsourced figures.
- Internal dashboards and watchdog tooling.

## License

Documentation and scripts here are provided as-is for reference, under the MIT
License (see [LICENSE](LICENSE)). Note that vLLM itself is Apache-2.0 and the
model weights carry their own license from their upstream author — respect both.
