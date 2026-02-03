import re
import os
import sys
import json
from dataclasses import dataclass, field
from pprint import pprint
from tqdm import tqdm
from collections import Counter

import torch
import numpy as np

from datasets import load_dataset
from evaluate import load

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, ".."))
)

from utils.gsm8k import extract_answer_number

from transformers import HfArgumentParser, set_seed, AutoModelForCausalLM, AutoTokenizer

import vllm
from vllm import SamplingParams

TEMPLATE = """
Your task is to answer the question below. Give step by step reasoning before you answer, and when you’re ready to answer, please use the format "The answer is: ..."\nQuestion: {question}
"""


@dataclass
class Arguments:
    dataset_name_or_path: str = field(default="gms8k")

    # model
    model_name_or_path: str = field(default="meta-llama/Llama-2-7b-chat")
    tokenizer_name_or_path: str = field(default="meta-llama/Llama-2-7b-chat")
    dtype: str = field(default="bf16", metadata={"choices": ["fp16", "bf16"]})

    # generation
    do_sample: bool = field(default=False)
    temperature: float = field(
        default=0.6,
    )
    top_k: int = field(default=50)
    top_p: float = field(default=1.0)
    n: int = field(default=1)

    use_vllm: bool = field(
        default=False, metadata={"help": "Whether use vLLM for generation."}
    )
    vllm_gpu_memory_utilization: float = field(
        default=0.9, metadata={"help": "vLLM GPU consumption ratio."}
    )

    seed: int = field(
        default=42, metadata={"help": "Random Seed for reproducing results."}
    )

    batch_size: int = field(default=16)
    max_new_tokens: int = field(default=512, metadata={"help": "Max response length."})

    # save
    remove_old: bool = field(
        default=False, metadata={"help": "Whether to remove old file."}
    )
    save_path: str = field(
        default="evaluation_gsm8k.json",
        metadata={"help": "Evaluation results save path."},
    )


def save_prompts_and_answers(
    model_name, prompts, labels, answers, evaluations, file_path
):
    assert len(prompts) == len(answers), "Mismatched lengths!"
    assert file_path.endswith(".json")
    data = [
        {
            "id": i,
            "model_name": model_name,
            "prompt": prompts[i],
            "label": labels[i],
            "answer": answers[i],
            "evaluation": evaluations[i],
        }
        for i in range(len(prompts))
    ]
    if not os.path.exists(file_path):
        with open(file_path, "w", encoding="utf-8") as file:
            json.dump(data, file, indent=2)
    else:
        with open(file_path, "r", encoding="utf-8") as file:
            data = json.load(file)

        # Determine the next id value
        next_id = data[-1]["id"] + 1 if data else 0

        # Create new entries and append them to the data list
        new_entries = [
            {
                "id": i + next_id,
                "model_name": model_name,
                "prompt": prompts[i],
                "label": labels[i],
                "answer": answers[i],
                "evaluation": evaluations[i],
            }
            for i in range(len(prompts))
        ]
        data.extend(new_entries)

        with open(file_path, "w", encoding="utf-8") as file:
            json.dump(data, file, indent=2)


def calculate_accuracy_voting(reference, candidates, depths=[1, 4, 8, 16, 32]):
    majority_voting_accuracies = []
    best_of_n_accuracies = []

    for depth in depths:
        if depth > len(candidates):
            break
        else:
            # Slice the candidates list to the current depth
            current_candidates = candidates[:depth]
            count = Counter(current_candidates)
            most_common = count.most_common(1)[0][0]  # Get the most frequent answer

            # Majority voting accuracy
            is_correct_majority = most_common == reference
            majority_voting_accuracies.append(is_correct_majority)

            # Best of n accuracy
            is_correct_best_of_n = reference in current_candidates
            best_of_n_accuracies.append(is_correct_best_of_n)

    return majority_voting_accuracies, best_of_n_accuracies


def main():
    parser = HfArgumentParser((Arguments,))
    (args,) = parser.parse_args_into_dataclasses()
    pprint(args.__dict__)

    if args.remove_old:
        if os.path.exists(args.save_path):
            os.remove(args.save_path)

    dataset = load_dataset(args.dataset_name_or_path, "main")
    dataset = dataset["test"]

    device = torch.device("cuda")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name_or_path)
    if "llama-3" in tokenizer.name_or_path.lower():
        tokenizer.pad_token = tokenizer.decode(len(tokenizer) - 1)
        tokenizer.pad_token_id = len(tokenizer) - 1
    elif "llama-2" in tokenizer.name_or_path.lower():
        tokenizer.pad_token = tokenizer.unk_token
        tokenizer.pad_token_id = tokenizer.unk_token_id
    # tokenizer.model_max_length = int(8096)
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}[args.dtype]

    if args.use_vllm:
        model = vllm.LLM(
            model=args.model_name_or_path,
            tokenizer=args.tokenizer_name_or_path,
            tensor_parallel_size=torch.cuda.device_count(),
            dtype=dtype,
            gpu_memory_utilization=args.vllm_gpu_memory_utilization,
            seed=args.seed,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            torch_dtype=dtype,
            attn_implementation="flash_attention_2",
        )
        model.to(device)
        model.eval()

    prompt_to_save = []
    ans_to_save = []
    labels_to_save = []
    evaluations_to_save = []

    count = 0
    max_depth = max(1, int(np.log2(args.n)))
    majority_voting_all = np.zeros([len(dataset), max_depth], dtype=int)
    best_of_n_all = np.zeros([len(dataset), max_depth], dtype=int)
    for i in tqdm(range(0, len(dataset), args.batch_size)):
        prompt = dataset[i : i + args.batch_size]["question"]
        prompt_conv = [
            [{"role": "user", "content": TEMPLATE.format(question=x)}] for x in prompt
        ]
        labels = [
            # x.replace("####", "Final answer:")
            x.replace("####", "The answer is:")
            for x in dataset[i : i + args.batch_size]["answer"]
        ]
        prompt_str = tokenizer.apply_chat_template(
            prompt_conv, tokenize=False, add_generation_prompt=True
        )

        tokenizer.padding_side = "left"
        prompt_token = tokenizer.apply_chat_template(
            prompt_conv,
            padding="longest",
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(device)

        prompt_length = prompt_token.input_ids.size(-1)

        if args.use_vllm:
            prompt_token_ids = [
                prompt_token.input_ids[
                    j, prompt_token.attention_mask[j].bool()
                ].tolist()
                for j in range(len(prompt_conv))
            ]

            sampling_params = SamplingParams(
                n=args.n,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                max_tokens=args.max_new_tokens,
            )
            with torch.no_grad():
                output_results = model.generate(
                    prompt_token_ids=prompt_token_ids, sampling_params=sampling_params
                )
            ans_str = []
            evaluation_results = []
            for j in range(len(output_results)):
                final_answers = []
                for k in range(args.n):
                    try:
                        answer = extract_answer_number(
                            output_results[j].outputs[k].text
                        )
                    except Exception as e:
                        print("========Error=========")
                        print(e)
                        print(output_results[j].outputs[k].text)
                        print()
                        answer = None
                    final_answers.append(answer)

                ans_str.append(
                    [output_results[j].outputs[k].text for k in range(args.n)]
                )
                true_answer = extract_answer_number(labels[j])
                majority_evaluation, best_of_n_evaluation = calculate_accuracy_voting(
                    true_answer, final_answers
                )
                majority_voting_all[count] = np.array(
                    majority_evaluation, dtype=np.int32
                )
                best_of_n_all[count] = np.array(best_of_n_evaluation, dtype=np.int32)
                count += 1

                evaluation_results.append(
                    {
                        "true_answer": true_answer,
                        "majority_evaluation": majority_evaluation,
                        "best_of_n_evaluation": best_of_n_evaluation,
                    }
                )

            prompt_to_save.extend(prompt_str)
            ans_to_save.extend(ans_str)
            labels_to_save.extend(labels)
            evaluations_to_save.extend(evaluation_results)

            pprint("===========Prompt=============")
            pprint(prompt_str[0])
            pprint("===========Label=============")
            pprint(labels[0])
            pprint("===========Response=============")
            pprint(ans_str[0])
            pprint("===========Evaluation=============")
            pprint(evaluation_results[0])
            pprint(
                "Majority Acc so far: {} Best Acc so far: {}".format(
                    np.round(np.mean(majority_voting_all[:count], axis=0) * 100, 2),
                    np.round(np.mean(best_of_n_all[:count], axis=0) * 100, 2),
                )
            )
        else:
            raise NotImplementedError

        if count % 128 == 0:
            save_prompts_and_answers(
                args.model_name_or_path,
                prompt_to_save,
                labels_to_save,
                ans_to_save,
                evaluations_to_save,
                args.save_path,
            )
            prompt_to_save.clear()
            ans_to_save.clear()
            labels_to_save.clear()
            evaluations_to_save.clear()

    if len(prompt_to_save) > 0:
        save_prompts_and_answers(
            args.model_name_or_path,
            prompt_to_save,
            labels_to_save,
            ans_to_save,
            evaluations_to_save,
            args.save_path,
        )
        prompt_to_save.clear()
        ans_to_save.clear()
        labels_to_save.clear()
        evaluations_to_save.clear()

    pprint(args.__dict__)
    print(
        "==> Majority Acc over the dataset: {} Best Acc over the dataset: {}".format(
            np.round(np.mean(majority_voting_all, axis=0) * 100, 2),
            np.round(np.mean(best_of_n_all, axis=0) * 100, 2),
        )
    )


if __name__ == "__main__":
    main()
