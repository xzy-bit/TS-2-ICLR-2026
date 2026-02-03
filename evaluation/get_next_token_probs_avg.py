import torch
import json
import argparse
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import torch.nn.functional as F

import os
import json
from pprint import pprint
from dataclasses import dataclass, field
from tqdm import tqdm
import pandas as pd

import vllm
from vllm import SamplingParams

import torch
from datasets import load_dataset, load_from_disk, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed, HfArgumentParser
from entmax import sparsemax
from utils.generate_utils import _apply_top_k_top_p
import numpy as np

@dataclass
class Arguments:
    model_name_or_path: str = field(
        default="meta-llama/Llama-2-7b-chat", metadata={"help": "Model name or path."}
    )

    tokenizer_path: str = field(
        default="meta-llama/Llama-2-7b-chat", metadata={"help": "Tokenizer path."}
    )

    max_size: int = field(
        default=None, metadata={"help": "Max data size for evaluation."}
    )

    use_sparsemax: bool = field(
        default=False, metadata={"help": "Whether use sparsemax for generation."}
    )

    seed: int = field(
        default=42, metadata={"help": "Random Seed for reproducing results."}
    )

    # generation
    batch_size: int = field(default=10)

    n: int = field(default=1, metadata={"help": "num of responses for each prompt."})
    do_sample: bool = field(
        default=True, metadata={"help": "Do sample for generation."}
    )
    top_k: int = field(default=50, metadata={"help": "Top k for generation."})
    top_p: float = field(default=0.9, metadata={"help": "Top p for generation."})
    temperature: float = field(
        default=0.6, metadata={"help": "Temperature for generation."}
    )
    max_new_tokens: int = field(default=1024, metadata={"help": "Max response length."})

    # save
    remove_old: bool = field(
        default=False, metadata={"help": "Whether to remove old file."}
    )
    save_path: str = field(
        default="evaluation_log_probability.json",
        metadata={"help": "Evaluation results save path."},
    )


def main():

    parser = HfArgumentParser((Arguments,))
    (args,) = parser.parse_args_into_dataclasses()
    pprint(args.__dict__)

    set_seed(args.seed)
    if os.path.exists(args.save_path):
        if args.remove_old:
            # if (
            #     input(
            #         "The given save path exists a file. Do you continue (yes ot no)?:\n"
            #     )
            #     != "yes"
            # ):
            #     assert 0
            os.remove(args.save_path)
        else:
            print("{} exists. Exit!".format(args.save_path))
            return

    device = torch.device("cuda")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    if "llama-3" in tokenizer.name_or_path.lower():
        tokenizer.pad_token = tokenizer.decode(len(tokenizer) - 1)
        tokenizer.pad_token_id = len(tokenizer) - 1
    elif "llama-2" in tokenizer.name_or_path.lower():
        tokenizer.pad_token = tokenizer.unk_token
        tokenizer.pad_token_id = tokenizer.unk_token_id

    prompts = ["Give me a single digit number",
                "Give me the name of a kinds of animal",
                "Give me a brand of a car",
                "Give me a brand of a phone",
                "Give me a name of a fruit"
              ]
    result = []

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    model.to(device)
    model.eval()

    results = np.zeros((5,20))
    for j in range(5):
        inputs = tokenizer(prompts[j], return_tensors="pt").to(device)
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits[:, -1, :]  # [B, vocab_size]

            # apply temperature
            if args.temperature != 1.0:
                logits = logits / args.temperature

            logits = logits.to(torch.float32)
            # logits = _apply_top_k_top_p(logits,args.top_k,args.top_p)
            # sparsemax
            sp_probs = sparsemax(logits, dim=-1)
            print(sp_probs.sum(-1))
            sf_probs = F.softmax(logits,dim=-1)
            print(sf_probs)

            if args.use_sparsemax:
                probs = sp_probs
            else:
                probs = sf_probs
            topk_probs, topk_indices = torch.topk(probs, k=20, dim=-1)
            result = []
            for i in range(20):
                prob = topk_probs[0,i].item()
                results[j][i] = prob

    #print(results[0])
    #print(results[1])
    means = np.mean(results,axis=0)
    for k in range(20):
        result.append({
            'rank':k,
            'prob':means[k]
        })
    # 保存为 JSON 文件
    with open(args.save_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved top 20 token probabilities to {args.save_path}")

if __name__ == "__main__":
    main()
