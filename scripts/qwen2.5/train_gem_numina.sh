#!/bin/bash

set -e 
set -x

export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export FLASH_ATTENTION_DETERMINISTIC="1"

TRAIN_TOKENIZED_FILE="./data/numina_sft_train_qwen2.5_tokenized.jsonl"
TEST_TOKENIZED_FILE="./data/numina_sft_test_qwen2.5_tokenized.jsonl"

MODEL_NAME_OR_PATH="Qwen/Qwen2.5-Math-7B"
SEED=1234

TIME_STEP=`date "+%Y-%m-%d-%H-%M-%S"`
OUTPUT_DIR="./log/sft_gem-qwen2.5_7b-numina-$TIME_STEP-$SEED"

mkdir -p $OUTPUT_DIR

deepspeed train.py \
    --deepspeed scripts/zero3.json \
    --seed $SEED \
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --train_tokenized_file $TRAIN_TOKENIZED_FILE \
    --test_tokenized_file $TEST_TOKENIZED_FILE \
    --output_dir $OUTPUT_DIR \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy "epoch" \
    --save_strategy "no" \
    --loss "gem" \
    --gem_beta 0.7 \
    --gem_h "logsigmoid" \
    --learning_rate 2e-5 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.03 \
    --num_train_epochs 3 \
    --logging_steps 10 \
    --report_to "tensorboard" \
    --gradient_checkpointing True \
    --overwrite_output_dir \
    --bf16 True \
    2>&1 | tee $OUTPUT_DIR/training.log