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

    prompt = "Give me a single digit number."
    result = []
    model = vllm.LLM(
        model=args.model_name_or_path,
        tokenizer=args.tokenizer_path,
        tensor_parallel_size=torch.cuda.device_count(),
        dtype=torch.bfloat16,
        gpu_memory_utilization=0.9,
        seed=args.seed,
        swap_space=16,
    )
    
    prompt_tokens = tokenizer(prompt)
    
    prompt_token_ids = [
            prompt_tokens.input_ids,
            ]
    sampling_params = SamplingParams(
                n=args.n,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                # stop_token_ids=[
                #     tokenizer.eos_token_id,
                #     tokenizer("<|eot_id|>").input_ids[-1],
                # ],
                logprobs=5,
                max_tokens=args.max_new_tokens,
            )

    with torch.no_grad():
        output_results = model.generate(prompt_token_ids=prompt_token_ids,sampling_params=sampling_params)
    
    token_stats=[]
    for j in range(len(output_results)):
        sample_outputs = []
        sample_logprobs = []

        for k in range(args.n):
            output = output_results[j].outputs[k]
            text = output.text
            logprobs_dict = output.logprobs
            
            for item in logprobs_dict:
                first_token_id, first_logprob_obj = next(iter(item.items()))
                token_stats.append({
                    "token":first_logprob_obj.decoded_token,
                    "logprob":first_logprob_obj.logprob,
                    "rank":first_logprob_obj.rank
                    })
    
    with open("token_logprobs.json", "w", encoding="utf-8") as f:
        json.dump(token_stats, f, indent=2, ensure_ascii=False)
    print("Saved to first_token_logprobs.json")
if __name__ == "__main__":
    main()
