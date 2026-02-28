import json

from tqdm import tqdm
import numpy as np
import re
import pandas as pd
from collections import Counter
import itertools
import os

def calculation_best_of_n_winrate(data):
    print("Calculating best of n winrate ....")
    best_n = np.zeros([len(data), 6])  # 1, 2, 4, 8, 16

    for i in tqdm(range(len(data))):
        rewards = data[i]["reward"]
        best_n[i][0] = rewards[0]>0
        best_n[i][1] = max(rewards[:2])>0
        best_n[i][2] = max(rewards[:4])>0
        best_n[i][3] = max(rewards[:8])>0
        best_n[i][4] = max(rewards[:16])>0
        best_n[i][5] = max(rewards[:32])>0
    length = len(best_n)
    best_n = best_n.sum(axis=0)*100/length
    for item in np.round(best_n,2):
        print(item,end='\t')
    print('\n')
    return best_n

def calculation_best_of_n_gpt(data,base_line_data):
    print("Calculating best of n reward ....")
    best_n = np.zeros([len(data), 6])  # 1, 2, 4, 8, 16
    base_line = np.zeros([len(data),1])
    for i in tqdm(range(len(data))):
        rewards = data[i]["reward"]
        best_n[i][0] = rewards[0]
        best_n[i][1] = max(rewards[:2])
        best_n[i][2] = max(rewards[:4])
        best_n[i][3] = max(rewards[:8])
        best_n[i][4] = max(rewards[:16])
        best_n[i][5] = max(rewards[:32])
        base_line[i][0] = base_line_data[i]["reward"]


    best_n_exp = np.exp(best_n)
    base_line_exp = np.exp(base_line)

    # print(best_n_exp)
    # print(base_line_exp)

    winrate_down = best_n_exp+base_line_exp
    winrate = best_n_exp/winrate_down

    winrate = winrate.mean(axis=0)
    for item in np.round(winrate*100,2):
        print(item,end='\t')
    return winrate

def calculation_best_of_n_baseline(data,base_line_data):
    print("Calculating best of n reward ....")
    best_n = np.zeros([len(data), 6])  # 1, 2, 4, 8, 16
    base_line = np.zeros([len(data),6])
    for i in tqdm(range(len(data))):
        rewards = data[i]["reward"]
        best_n[i][0] = rewards[0]
        best_n[i][1] = max(rewards[:2])
        best_n[i][2] = max(rewards[:4])
        best_n[i][3] = max(rewards[:8])
        best_n[i][4] = max(rewards[:16])
        best_n[i][5] = max(rewards[:32])

        base_line_rewards = base_line_data[i]["reward"]
        base_line[i][0] = base_line_rewards[0]
        base_line[i][1] = max(base_line_rewards[:2])
        base_line[i][2] = max(base_line_rewards[:4])
        base_line[i][3] = max(base_line_rewards[:8])
        base_line[i][4] = max(base_line_rewards[:16])
        base_line[i][5] = max(base_line_rewards[:32])

    best_n_exp = np.exp(best_n)
    base_line_exp = np.exp(base_line)

    # print(best_n_exp)
    # print(base_line_exp)

    winrate_down = best_n_exp+base_line_exp
    winrate = best_n_exp/winrate_down

    print(winrate)
    winrate = winrate.mean(axis=0)
    for item in np.round(winrate*100,2):
        print(item,end='\t')
    return winrate

def calculation_mean_of_n(data):
    print("Calculating best of n reward ....")
    best_n = np.zeros([len(data), 6])  # 1, 2, 4, 8, 16

    for i in tqdm(range(len(data))):
        rewards = data[i]["reward"]
        best_n[i][0] = rewards[0]
        best_n[i][1] = sum(rewards[:2])/2.0
        best_n[i][2] = sum(rewards[:4])/4.0
        best_n[i][3] = sum(rewards[:8])/8.0
        best_n[i][4] = sum(rewards[:16])/16.0
        best_n[i][5] = sum(rewards[:32])/32.0

    best_n = best_n.mean(axis=0)
    for item in np.round(best_n,2):
        print(item,end='\t')
    print('\n')
    return best_n


def analyze_response(data,csv_path):
    for i in tqdm(range(len(data))):
        # if i==5:
        #     break

        # rewards = data[i]["reward"]
        ranks = data[i]["ranks"]
        logprobs = data[i]["logprobs"]

        for idx in range(len(ranks)):
            rank = np.array(ranks[idx])
            logprob = np.array(logprobs[idx])
            # reward = rewards[idx]
            # print(reward)
            # print(rank.mean())
            # print(np.exp(logprob).mean())
            data_to_save = {

                # "reward": [reward],
                "avg_rank": [rank.mean()],
                "avg_prob": [np.exp(logprob).mean()]
            }

            df = pd.DataFrame(data_to_save)

            df.to_csv(csv_path, mode="a+", index=False, header=not os.path.exists(csv_path))

def main():
    data_path = "sft_hybrid_llama-3.1-8b_sparsemax_infer_logprob_alpaca_eval-seed_42-n_32-T_0.6-K_50-P_0.9-reward.json"
    response_data = json.load(open(data_path, "r",encoding='utf-8'))


    # base_line = "GPT-4-reward.json"
    # base_line_data = json.load(open(base_line,"r",encoding='utf-8'))
    # calculation_best_of_n_gpt(response_data,base_line_data)


    # base_line = "sft_ce_llama-3.1-8b_alpaca_eval-seed_42-n_32-T_0.6-K_50-P_0.9-reward.json"
    # base_line_data = json.load(open(base_line,"r",encoding='utf-8'))
    # calculation_best_of_n_baseline(response_data,base_line_data)

if __name__ == "__main__":
    # analyze_raw_json_text(data_path)
    main()
