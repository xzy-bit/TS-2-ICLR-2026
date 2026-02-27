#!/bin/sh

set -e
set -x

export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export CUDA_VISIBLE_DEVICES=2

DATA_PATH="tatsu-lab/alpaca_eval"
MODEL_NAME="sft_ts2_llama31-8b"

# Test script
MODEL_PATH="meta-llama/Llama-3.1-8B-Instruct"

#MODEL_PATH="./log/sft_ts2-llama-3.1_8b-ultrafeedback-2025-08-23-15-49-31-1234"
TOKENIZER_PATH="meta-llama/Llama-3.1-8B-Instruct"
REWARD_MODEL="sfairXC/FsfairX-LLaMA3-RM-v0.1"

RESPONSE_PATH="alpacal_eval/response"
SAVED_PATH="alpacal_eval/winrate"

mkdir -p alpacal_eval
mkdir -p alpacal_eval/response
mkdir -p alpacal_eval/results

SEED=42
T=0.6
K=50
P=0.9
N=32


python  evaluation/generate_response.py \
    --model_name_or_path $MODEL_PATH \
    --tokenizer_path $TOKENIZER_PATH \
    --dataset_path $DATA_PATH \
    --max_size 1000 \
    --seed $SEED \
    --temperature $T \
    --top_k $K \
    --top_p $P \
    --max_new_tokens 2048 \
    --n $N \
    --use_vllm True \
    --save_path "${RESPONSE_PATH}/${MODEL_NAME}_alpaca_eval-seed_${SEED}-n_${N}-T_${T}-K_${K}-P_${P}.json"

python evaluation/evaluation_reward.py \
    --model_name_or_path $REWARD_MODEL \
    --batch_size 4 \
    --detokenizer_path $TOKENIZER_PATH \
    --data_path "${RESPONSE_PATH}/${MODEL_NAME}_alpaca_eval-seed_${SEED}-n_${N}-T_${T}-K_${K}-P_${P}.json" \
    --save_path "${RESPONSE_PATH}/${MODEL_NAME}_alpaca_eval-seed_${SEED}-n_${N}-T_${T}-K_${K}-P_${P}-reward.json"  \
    2>&1 | tee "${SAVED_PATH}/${MODEL_NAME}_reward_eval-seed_${SEED}-n_${N}-T_${T}-K_${K}-P_${P}.log"

python evaluation/evaluation_diversity.py \
    --tokenizer_path $TOKENIZER_PATH \
    --detokenizer_path $TOKENIZER_PATH \
    --response_path "${RESPONSE_PATH}/${MODEL_NAME}_alpaca_eval-seed_${SEED}-n_${N}-T_${T}-K_${K}-P_${P}.json"\
    2>&1 | tee ${SAVED_PATH}/${MODEL_NAME}_diversity_eval-alpaca_eval-seed_${SEED}-n_${N}-T_${T}_K_${K}_P_${P}.log
