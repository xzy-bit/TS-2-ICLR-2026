set -euo pipefail
set -x

# --- Inputs (REQUIRED via env) ---
: "${MODEL_DIR:?set via env: MODEL_DIR=/path/to/model}"
: "${RUN_NAME:?set via env: RUN_NAME=name_for_outputs}"
: "${TOKENIZER_PATH:?set via env: TOKENIZER_PATH=/path/to/tokenizer}"
: "${TASK:?set via env: TASK=arc|hellaswag|winogrande|mmlu|truthfulqa}"

# --- Sane defaults ---
OUT_BASE="./evals_openllm/$RUN_NAME/$TASK"
TEMPERATURE="${TEMPERATURE:-0.6}"
TOP_P="${TOP_P:-0.9}"
TOP_K="${TOP_K:--1}"               # -1 disables top-k in vLLM 0.7.1
MAX_NEW="${MAX_NEW:-256}"
N="${N:-32}"
BATCH_SIZE="${BATCH_SIZE:-16}"
USE_VLLM="${USE_VLLM:-True}"

mkdir -p "$OUT_BASE"
DUMP="$OUT_BASE/${TASK}_voting_dump.json"
SUMMARY="$OUT_BASE/${TASK}_voting_summary.json"
LOG="$OUT_BASE/${TASK}_voting.log"

# --- Run ---
python -u openllm_evaluation.py \
  --task "$TASK" \
  --model_name_or_path "$MODEL_DIR" \
  --tokenizer_name_or_path "$TOKENIZER_PATH" \
  --dtype bf16 \
  --temperature "$TEMPERATURE" \
  --top_p "$TOP_P" \
  --top_k "$TOP_K" \
  --n "$N" \
  --batch_size "$BATCH_SIZE" \
  --max_new_tokens "$MAX_NEW" \
  --use_vllm "$USE_VLLM" \
  --vllm_gpu_memory_utilization 0.9 \
  --remove_old True \
  --save_path "$DUMP" \
  --summary_path "$SUMMARY" \
  2>&1 | tee "$LOG"

echo "[done] wrote:"
ls -lh "$OUT_BASE"
