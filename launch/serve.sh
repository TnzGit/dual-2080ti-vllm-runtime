#!/usr/bin/env bash
# Launch the dual RTX 2080 Ti 22GB vLLM runtime (text-only, 230K ctx, MTP=5).
#
# Set the paths below to match your own environment before running.
set -euo pipefail

# --- adjust these -----------------------------------------------------------
MODEL_PATH="${MODEL_PATH:-/path/to/your/orcarouter-Qwen3.8-27B-Uncensored-NVFP4}"
VLLM_BIN="${VLLM_BIN:-vllm}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
# ---------------------------------------------------------------------------

SERVED_NAME="qwen38-27b-orcarouter-nvfp4-fp8kv-230K-mtp5-text-only-cu130"

# Profile-derived values (see ../profiles/orcarouter-fp8kv-230K-mtp5-text-only.env)
export PYTORCH_CUDA_ALLOC_CONF="garbage_collection_threshold:0.8"
export VLLM_QWOPUS_MTP_BF16_DRAFT=1
export VLLM_ALLOW_MAMBA_SPEC_FULL_CUDAGRAPH=0
export VLLM_TURBOQUANT_CONTINUATION_PREFIX_COMBINE=auto
export VLLM_TURBOQUANT_CONTINUATION_PREFIX_COMBINE_MIN_TOKENS=20480
export VLLM_TURBOQUANT_DECODE_BLOCK_KV=2

exec "$VLLM_BIN" serve "$MODEL_PATH" \
  --host "$HOST" \
  --port "$PORT" \
  --served-model-name "$SERVED_NAME" \
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
