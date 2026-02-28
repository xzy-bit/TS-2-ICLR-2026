#!/bin/bash

set -e 
set -x

export CUDA_VISIBLE_DEVICES=3

MODEL_PATH="./log/sft_ts2-llama3.1-8b-ultrafeedback-2026-02-27-23-31-31-1234"
#MODEL_PATH="meta-llama/Llama-3.1-8B-Instruct"
MODEL_NAME="sft_ts2"
TOKENIZER_PATH="meta-llama/Llama-3.1-8B-Instruct"

TASKS="arc hellaswag winogrande mmlu truthfulqa gsm8k"

for TASK in $TASKS; do
  echo ">>> Running task: $TASK"
  MODEL_DIR=$MODEL_PATH RUN_NAME=$MODEL_NAME TOKENIZER_PATH=$TOKENIZER_PATH TASK=$TASK \
    bash ./voting_openllm.sh

done

echo "Job finished on $(date)"
python ./utils/print_openllm_results.py
