#!/bin/sh

# set -e 
set -x

export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export CUDA_VISIBLE_DEVICES="0"

MODEL_KEY="bigcode/starcoder"
MODEL_NAME="sft_ts2-llama-3.1_8b"
#MODEL_PATH="./log/sft_ts2-llama-3.1_8b-ultrafeedback-2025-08-23-15-49-31-1234"

# Testing evaluation

MODEL_PATH="meta-llama/Llama-3.1-8B-Instruct"
TOKENIZER_PATH="meta-llama/Llama-3.1-8B-Instruct"
METHOD_NAME="ts2"

DATASET="humaneval"
RESPONSE_PATH="humaneval_results/response/${MODEL_NAME}_evalplus-$DATASET.jsonl"
SAVE_PATH="humaneval_results/"

mkdir -p humaneval_results/response
mkdir -p humaneval_results

python -m evaluation.text2code \
  --model_key $MODEL_KEY \
  --model_name_or_path $MODEL_PATH \
  --tokenizer_name_or_path $TOKENIZER_PATH \
  --save_path $RESPONSE_PATH \
  --dataset $DATASET \
  --temperature 0.6 \
  --top_p 0.9 \
  --max_new_tokens 512 \
  --n_problems_per_batch 100 \
  --n_samples_per_problem 200 \
  --n_batches 1 \
  --use_vllm True \
  --template normal 

evalplus.evaluate --dataset $DATASET --samples $RESPONSE_PATH\
    2>&1 | tee $SAVE_PATH/${MODEL_NAME}_humaneval.log

