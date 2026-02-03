#!/bin/bash

set -e 
set -x

export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export FLASH_ATTENTION_DETERMINISTIC="1"
export CUDA_VISIBLE_DEVICES="0"

# tokenize train data
python preprocess_data.py \
    --dataset_name_or_path "AIMO/NuminaMath-CoT" \
    --split "train" \
    --tokenizer_name_or_path "Qwen/Qwen2.5-Math-7B" \
    --max_seq_length 2048 \
    --output_file "./data/numina_sft_train_qwen2.5_tokenized.jsonl" \
    --start 0 \
    --end 20000 

# tokenize test data 
python preprocess_data.py \
    --dataset_name_or_path "AIMO/NuminaMath-CoT" \
    --split "train" \
    --tokenizer_name_or_path "Qwen/Qwen2.5-Math-7B" \
    --max_seq_length 2048 \
    --output_file "./data/numina_sft_test_qwen2.5_tokenized.jsonl" \
    --start 20000 \
    --end 21000 