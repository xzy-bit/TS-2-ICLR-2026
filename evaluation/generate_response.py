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

    # dataset
    dataset_path: str = field(
        default="RLHFlow/Orca-distibalel-standard", metadata={"help": "Dataset path."}
    )
    split: str = field(default=None, metadata={"help": "Split of the dataset."})
    column_name: str = field(
        default=None, metadata={"help": "Column name to extract prompt."}
    )
    standard_format: bool = field(
        default=None, metadata={"help": "Dataset in the standard format."}
    )
    load_from_disk: bool = field(
        default=False, metadata={"help": "Whether use the load_from_disk method."}
    )
    max_size: int = field(
        default=None, metadata={"help": "Max data size for evaluation."}
    )

    use_vllm: bool = field(
        default=False, metadata={"help": "Whether use vLLM for generation."}
    )
    vllm_gpu_memory_utilization: float = field(
        default=0.9, metadata={"help": "vLLM GPU consumption ratio."}
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

    def __post_init__(self):
        if self.column_name is None:
            if "tatsu-lab/alpaca_eval" in self.dataset_path:
                self.column_name = "instruction"
            if "HuggingFaceH4/ultrachat_200k" in self.dataset_path:
                self.column_name = "prompt"
            if "if_eval" in self.dataset_path:
                self.column_name = "prompt"
            if "poem_generation" in self.dataset_path:
                self.column_name = "instruction"
            if "story_generation" in self.dataset_path:
                self.column_name = "instruction"

        if self.split is None:
            if "tatsu-lab/alpaca_eval" in self.dataset_path:
                self.split = "eval"
            if "HuggingFaceH4/ultrachat_200k" in self.dataset_path:
                self.split = "test_sft"
            if "if_eval" in self.dataset_path:
                self.split = "train"
            if "poem_generation" in self.dataset_path:
                self.split = "test"
            if "story_generation" in self.dataset_path:
                self.split = "test"

        if self.standard_format is None:
            if "tatsu-lab/alpaca_eval" in self.dataset_path:
                self.standard_format = False
            if "HuggingFaceH4/ultrachat_200k" in self.dataset_path:
                self.standard_format = False
            if "if_eval" in self.dataset_path:
                self.standard_format = False
            if "poem_generation" in self.dataset_path:
                self.standard_format = False
            if "story_generation" in self.dataset_path:
                self.standard_format = False


def get_dataset(dataset_name, split="test", from_disk=False):
    if from_disk:
        dataset = load_from_disk(dataset_name)
    else:
        if "tatsu-lab/alpaca_eval" in dataset_name:
            dataset = load_dataset(dataset_name, "alpaca_eval")
        if "if_eval" in dataset_name:
            dataset = []
            with open("./data/if_eval_data.jsonl") as f:
                for line in f.readlines():
                    dataset.append(json.loads(line))
            dataset = Dataset.from_pandas(pd.DataFrame(dataset))
            return dataset
        else:
            dataset = load_dataset(dataset_name)
    if split in dataset:
        return dataset[split]
    else:
        assert "train" in dataset
        total_size = len(dataset["train"])
        eval_size = min(1000, int(total_size * 0.1))
        train_size = total_size - eval_size
        print(
            "There is no {} in the dataset. I set {} samples from the train split.".format(
                split, eval_size
            )
        )
        return dataset["train"].shuffle(seed=42).select(range(train_size, total_size))


def save_prompts_and_answers(model_name, prompts, answers,ranks,logprobs, file_path):
    assert len(prompts) == len(answers), "Mismatched lengths!"
    assert file_path.endswith(".json")
    data = [
        {
            "id": i,
            "model_name": model_name,
            "prompt": prompts[i],
            "answer": answers[i],
            "ranks": ranks[i],
            "logprobs": logprobs[i]    
        }
        for i in range(len(prompts))
    ]
    if not os.path.exists(file_path):
        with open(file_path, "w", encoding="utf-8") as file:
            json.dump(data, file, separators=(",", ":"), ensure_ascii=False)
    else:
        with open(file_path, "r", encoding="utf-8") as file:
            data = json.load(file)

        # Determine the next id value
        next_id = data[-1]["id"] + 1 if data else 0

        # Create new entries and append them to the data list
        new_entries = [
            {
                "id": next_id + i,
                "model_name": model_name,
                "prompt": prompts[i],
                "answer": answers[i],
                "ranks": ranks[i],
                "logprobs": logprobs[i],
            }
            for i in range(len(prompts))
        ]
        data.extend(new_entries)

        with open(file_path, "w", encoding="utf-8") as file:
            json.dump(data, file, separators=(",", ":"), ensure_ascii=False)


def main():
    parser = HfArgumentParser((Arguments,))
    (args,) = parser.parse_args_into_dataclasses()
    pprint(args.__dict__)

    training_config = {}
    if os.path.exists(os.path.join(args.model_name_or_path, "args.json")) and False:
        # f = open(os.path.join(args.model_name_or_path, "args.json"), "r")
        # for line in f.readlines()[:-1]:
        #     line_dict = json.loads(line)
        #     for key, val in line_dict.items():
        #         training_config[key] = val
        training_config = json.load(
            open(os.path.join(args.model_name_or_path, "args.json"), "r")
        )
        if training_config["algo"] == "sft":
            key_parameters = {
                "algo": "sft",
                "model_name_or_path": training_config["model_name_or_path"],
                "dataset": training_config["data_path"],
                "data_max_size": training_config["max_size"],
                "learning_rate": training_config["learning_rate"],
                "num_train_epochs": (training_config["num_train_epochs"]),
                "max_seq_len": training_config["max_seq_len"],
            }
        elif training_config["algo"] == "dpo":
            key_parameters = {
                "algo": "dpo",
                "actor_model_name_or_path": training_config["actor_model_name_or_path"],
                "max_entropy": training_config["max_entropy"],
                "beta": training_config["beta"],
                "tau": training_config["tau"] if "tau" in training_config else None,
                "gamma": training_config["gamma"],
                "alpha": training_config["alpha"],
                "dataset": training_config["data_path"],
                "learning_rate": training_config["actor_learning_rate"],
                "enable_ema": (
                    training_config["enable_ema"]
                    if "enable_ema" in training_config
                    else None
                ),
                "ema_coeff": (
                    training_config["ema_coeff"]
                    if "ema_coeff" in training_config
                    else None
                ),
            }
        print("===========Your Training Key Parameters===============")
        pprint(key_parameters)
        print("Your save path: {}".format(args.save_path))
        print()

        # if input("Is the save path correct (yes or no)?:\n") != "yes":
        #     assert 0
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

    dataset = get_dataset(args.dataset_path, args.split, args.load_from_disk)
    if args.max_size:
        dataset = dataset.select(range(0, min(len(dataset), args.max_size)))

    device = torch.device("cuda")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    if "llama-3" in tokenizer.name_or_path.lower():
        tokenizer.pad_token = tokenizer.decode(len(tokenizer) - 1)
        tokenizer.pad_token_id = len(tokenizer) - 1
    elif "llama-2" in tokenizer.name_or_path.lower():
        tokenizer.pad_token = tokenizer.unk_token
        tokenizer.pad_token_id = tokenizer.unk_token_id

    if args.use_vllm:
        model = vllm.LLM(
            model=args.model_name_or_path,
            tokenizer=args.tokenizer_path,
            tensor_parallel_size=torch.cuda.device_count(),
            dtype=torch.bfloat16,
            gpu_memory_utilization=args.vllm_gpu_memory_utilization,
            seed=args.seed,
            swap_space=16,
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

    # eos_token_id = [tokenizer.eos_token_id]
    # if "llama-3" in tokenizer.name_or_path.lower():
    #     eos_token_id.append(tokenizer("<|eot_id|>").input_ids[-1])

    prompt_to_save = []
    ans_to_save = []
    logprobs_to_save = []
    ranks_to_save = []
    count = 0
    for i in tqdm(range(0, len(dataset), args.batch_size)):
        if args.standard_format:
            chosen_conv = dataset[i : i + args.batch_size][args.column_name]
            prompt_conv = [x[:-1] for x in chosen_conv]

            prompt_str = tokenizer.apply_chat_template(
                prompt_conv, tokenize=False, add_generation_prompt=True
            )
        else:
            prompt = dataset[i : i + args.batch_size][args.column_name]
            prompt_conv = [[{"role": "user", "content": x}] for x in prompt]

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
                output_results = model.generate(
                    prompt_token_ids, sampling_params=sampling_params
                )
            ans_str = []
            logprobs = []
            ranks = []
            
            for j in range(len(output_results)):
                sample_outputs = []
                sample_logprobs = []
                sample_ranks = []
                for k in range(args.n):
                    output = output_results[j].outputs[k]
                    sample_outputs.append(output.text)
                    logprobs_dict = output.logprobs
                    
                    text_probs = []
                    text_ranks = []
                    for item in logprobs_dict:
                        ids,value = next(iter(item.items()))
                        text_probs.append(value.logprob)
                        text_ranks.append(value.rank)

                    sample_logprobs.append(text_probs)
                    sample_ranks.append(text_ranks)

                ans_str.append(sample_outputs)
                logprobs.append(sample_logprobs)
                ranks.append(sample_ranks)

            prompt_to_save.extend(prompt_str)
            ans_to_save.extend(ans_str)
            logprobs_to_save.extend(logprobs)
            ranks_to_save.extend(ranks)
            print(ranks[0])
        else:
            with torch.no_grad():
                outputs = model.generate(
                    prompt_token.input_ids,
                    attention_mask=prompt_token.attention_mask,
                    top_k=args.top_k,
                    top_p=args.top_p,
                    temperature=args.temperature,
                    max_new_tokens=args.max_new_tokens,
                    pad_token_id=tokenizer.pad_token_id,
                    #  eos_token_id=eos_token_id,
                    do_sample=args.do_sample,
                )
            ans_token = outputs[:, prompt_length:]
            ans_str = tokenizer.batch_decode(ans_token, skip_special_tokens=True)
            # ans_str = [x.replace("<|eot_id|>", "") for x in ans_str]

            prompt_to_save.extend(prompt_str)
            ans_to_save.extend(ans_str)
            count += 1

            print(prompt_str[0])
            print(ans_str[0])

        if count % 10 == 0:
            save_prompts_and_answers(
                args.model_name_or_path,
                prompt_to_save,
                ans_to_save,
                ranks_to_save,
                logprobs_to_save,
                args.save_path,
            )
            prompt_to_save.clear()
            ans_to_save.clear()

    if len(prompt_to_save) > 0:
        save_prompts_and_answers(
            args.model_name_or_path,
            prompt_to_save,
            ans_to_save,
            ranks_to_save,
            logprobs_to_save,
            args.save_path,
        )
        prompt_to_save.clear()
        ans_to_save.clear()


if __name__ == "__main__":
    main()
