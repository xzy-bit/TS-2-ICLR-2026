import re
import os
import sys
import json
from dataclasses import dataclass, field
from pprint import pprint
from tqdm import tqdm
from fraction import Fraction

import torch
import numpy as np

from datasets import load_dataset
# from evaluate import load

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

    # generation
    do_sample: bool = field(default=False)

    use_vllm: bool = field(
        default=False, metadata={"help": "Whether use vLLM for generation."}
    )
    vllm_gpu_memory_utilization: float = field(
        default=0.9, metadata={"help": "vLLM GPU consumption ratio."}
    )

    seed: int = field(
        default=42, metadata={"help": "Random Seed for reproducing results."}
    )

    batch_size: int = field(default=10)
    top_k: int = field(default=-1)
    top_p: float = field(default=1.0)
    temperature: float = field(default=0.0, metadata={"help": "Temperature."})
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

    if args.use_vllm:
        model = vllm.LLM(
            model=args.model_name_or_path,
            tokenizer=args.tokenizer_name_or_path,
            tensor_parallel_size=torch.cuda.device_count(),
            dtype=torch.bfloat16,
            gpu_memory_utilization=args.vllm_gpu_memory_utilization,
            seed=args.seed,
        )
        args.batch_size = len(dataset)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        )
        model.to(device)
        model.eval()

    prompt_to_save = []
    ans_to_save = []
    labels_to_save = []
    evaluations_to_save = []
    count = 0
    total_acc = 0
    total_num = 0
    for i in tqdm(range(0, len(dataset), args.batch_size)):
        prompt = dataset[i : i + args.batch_size]["question"]
        prompt_conv = [
            [{"role": "user", "content": TEMPLATE.format(question=x)}] for x in prompt
        ]
        labels = [
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
                top_k=args.top_k,
                top_p=args.top_p,
                temperature=args.temperature,
                max_tokens=args.max_new_tokens,
            )
            with torch.no_grad():
                output_results = model.generate(
                    prompt_token_ids=prompt_token_ids, sampling_params=sampling_params
                )
            ans_str = []
            for j in range(len(output_results)):
                ans_str.append(output_results[j].outputs[0].text)

            evaluation_results = []
            for j in range(len(output_results)):
                try:
                    answer = extract_answer_number(ans_str[j])
                except Exception as e:
                    print("========Error=========")
                    print(e)
                    print(ans_str[i])
                    print()
                    answer = None
                true_answer = extract_answer_number(labels[j])
                if answer is not None:
                    evaluation_results.append(
                        (answer, true_answer, answer == true_answer)
                    )
                else:
                    evaluation_results.append((answer, true_answer, None))

            total_num += len([x for x in evaluation_results if x[-1] is not None])
            total_acc += len(
                [x for x in evaluation_results if x is not None and x[-1] is True]
            )

            prompt_to_save.extend(prompt_str)
            ans_to_save.extend(ans_str)
            labels_to_save.extend(labels)
            evaluations_to_save.extend(evaluation_results)
            count += 1

            print("===========Prompt=============")
            print(prompt_str[0])
            print("===========Label=============")
            print(labels[0])
            print("===========Response=============")
            print(ans_str[0])
            print("===========Evaluation=============")
            print(evaluation_results[0])
        else:
            with torch.no_grad():
                outputs = model.generate(
                    prompt_token.input_ids,
                    attention_mask=prompt_token.attention_mask,
                    max_new_tokens=args.max_new_tokens,
                    pad_token_id=tokenizer.pad_token_id,
                    do_sample=args.do_sample,
                )
            ans_token = outputs[:, prompt_length:]
            ans_str = tokenizer.batch_decode(ans_token, skip_special_tokens=True)

            evaluation_results = []
            for j in range(len(output_results)):
                try:
                    answer = extract_answer_number(ans_str[j])
                except Exception as e:
                    print("========Error=========")
                    print(e)
                    print(ans_str[i])
                    print()
                    answer = None
                true_answer = extract_answer_number(labels[j])
                if answer is not None:
                    evaluation_results.append(
                        (answer, true_answer, answer == true_answer)
                    )
                else:
                    evaluation_results.append((answer, true_answer, None))

            total_num += len([x for x in evaluation_results if x[-1] is not None])
            total_acc += len(
                [x for x in evaluation_results if x is not None and x[-1] is True]
            )

            prompt_to_save.extend(prompt_str)
            ans_to_save.extend(ans_str)
            labels_to_save.extend(labels)
            evaluations_to_save.extend(evaluation_results)
            count += 1

            print("===========Prompt=============")
            print(prompt_str[0])
            print("===========Label=============")
            print(labels[0])
            print("===========Response=============")
            print(ans_str[0])
            print("===========Evaluation=============")
            print(evaluation_results[0])

        if count % 10 == 0:
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

    if total_num > 0:
        pprint(args.__dict__)
        print(
            "Acc over {} valid answers is {:.4f}, over {} all answers is {:.4f}".format(
                total_num, total_acc / total_num, len(dataset), total_acc / len(dataset)
            )
        )


if __name__ == "__main__":
    main()
