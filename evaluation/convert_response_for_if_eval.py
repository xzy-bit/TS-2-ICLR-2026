import os
import json
from pprint import pprint
from tqdm import tqdm
import pandas as pd

from dataclasses import dataclass, field
from transformers import AutoTokenizer, HfArgumentParser
from datasets import load_dataset, Dataset


@dataclass
class Arguments:
    response_path: str = field(
        default=None,
        metadata={"help": "Response path (json file) to convert."},
    )
    tokenizer_path: str = field(
        default="meta-llama/Meta-Llama-3-8B-Instruct",
        metadata={"help": "Tokenizer path to help clean str."},
    )
    save_path: str = field(default="alpaca_eval_response.json")


def main():
    parser = HfArgumentParser((Arguments,))
    (args,) = parser.parse_args_into_dataclasses()

    pprint(args.__dict__)

    old_data = json.load(open(args.response_path, "r"))

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)

    dataset = []
    with open("./instruction_following_eval/data/input_data.jsonl") as f:
        for line in f.readlines():
            dataset.append(json.loads(line))
    if_eval_dataset = Dataset.from_pandas(pd.DataFrame(dataset))

    new_data = []

    for i in tqdm(range(len(old_data))):
        prompt = old_data[i]["prompt"]

        prompt_clean = (
            tokenizer.decode(
                tokenizer(prompt.replace(tokenizer.bos_token, "")).input_ids,
                skip_special_tokens=True,
            )
            .replace("user\n\n", "")
            .replace("assistant\n\n", "")
        )
        prompt_ref = if_eval_dataset[i]["prompt"]

        if prompt_clean.strip()[:10] != prompt_ref.strip()[:10]:
            import ipdb

            ipdb.set_trace()

        new_data.append(
            {
                "id": i,
                "prompt": prompt_ref,
                "response": (
                    old_data[i]["answer"]
                    if isinstance(old_data[i]["answer"], str)
                    else old_data[i]["answer"][0]
                    .replace("<|eot_id|>", "")
                    .replace(tokenizer.eos_token, "")
                    .strip()
                ),
                "generator": old_data[i]["model_name"],
            }
        )
    os.makedirs(
        args.save_path.replace(args.save_path.split("/")[-1], ""), exist_ok=True
    )

    with open(args.save_path, "w") as outfile:
        for entry in new_data:
            json.dump(entry, outfile)
            outfile.write("\n")
    print(f"Save response to {args.save_path}")


if __name__ == "__main__":
    main()
