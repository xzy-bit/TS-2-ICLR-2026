import json
import os

os.environ["TOKENIZERS_PARALLELISM"] = "true"
import torch

from datasets import load_dataset

from argparse import ArgumentParser
from transformers import AutoTokenizer
from multiprocessing import Pool
from tqdm import tqdm

parser = ArgumentParser()
parser.add_argument(
    "--dataset_name_or_path",
    type=str,
    default="HuggingFaceH4/ultrafeedback_binarized",
)
parser.add_argument(
    "--split",
    type=str,
    default="train",
)
parser.add_argument(
    "--start",
    type=int,
    default=0,
)
parser.add_argument(
    "--end",
    type=int,
    default=None,
)
parser.add_argument(
    "--output_file",
    type=str,
)
parser.add_argument(
    "--tokenizer_name_or_path",
    type=str,
    required=True
)
parser.add_argument("--sft_path",type=str)
parser.add_argument("--usft_path",type=str)
parser.add_argument("--max_seq_length", type=int, default=4096)
parser.add_argument("--preprocessing_num_workers", type=int, default=64)
parser.add_argument("--proportion", type=float, default=1.0)
args = parser.parse_args()

tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name_or_path)
print(f"load tokenizer from {args.tokenizer_name_or_path} done.")
max_seq_length = args.max_seq_length
proportion = args.proportion

input_data = load_dataset(args.dataset_name_or_path)
if args.split:
    input_data = input_data[args.split]
if args.end is None:
    args.end = len(input_data)
input_data = input_data.select(range(args.start, args.end)).shuffle(seed=42)
print(
    f"load input data from {args.dataset_name_or_path} done. len(input_data): {len(input_data)}"
)


def encode_sft_example(example, verbose=False):
    """
    This function encodes a single example into a format that can be used for sft training.
    Here, we assume each example has a 'messages' field. Each message in it is a dict with 'role' and 'content' fields.
    We use the `apply_chat_template` function from the tokenizer to tokenize the messages and prepare the input and label tensors.
    """
    messages = example["messages"]
    if len(messages) == 0:
        raise ValueError("messages field is empty.")
    if verbose:
        chat_messages = tokenizer.apply_chat_template(
            conversation=messages,
            tokenize=False,
            return_tensors="pt",
            padding=False,
            truncation=True,
            max_length=max_seq_length,
            add_generation_prompt=False,
        )
        print(f"chat_messages:\n[{chat_messages}]")
    input_ids = tokenizer.apply_chat_template(
        conversation=messages,
        tokenize=True,
        return_tensors="pt",
        padding=False,
        truncation=True,
        max_length=max_seq_length,
        add_generation_prompt=False,
    )
    labels = input_ids.clone()
    # mask the non-assistant part for avoiding loss
    for message_idx, message in enumerate(messages):
        if message["role"] != "assistant":
            # we calculate the start index of this non-assistant message
            if message_idx == 0:
                message_start_idx = 0
            else:
                message_start_idx = tokenizer.apply_chat_template(
                    conversation=messages[
                        :message_idx
                    ],  # here marks the end of the previous messages
                    tokenize=True,
                    return_tensors="pt",
                    padding=False,
                    truncation=True,
                    max_length=max_seq_length,
                    add_generation_prompt=False,
                ).shape[1]
            # next, we calculate the end index of this non-assistant message
            if (
                message_idx < len(messages) - 1
                and messages[message_idx + 1]["role"] == "assistant"
            ):
                # for intermediate messages that follow with an assistant message, we need to
                # set `add_generation_prompt=True` to avoid the assistant generation prefix being included in the loss
                # (e.g., `<|assistant|>`)
                message_end_idx = tokenizer.apply_chat_template(
                    conversation=messages[: message_idx + 1],
                    tokenize=True,
                    return_tensors="pt",
                    padding=False,
                    truncation=True,
                    max_length=max_seq_length,
                    add_generation_prompt=True,
                ).shape[1]
            else:
                # for the last message or the message that doesn't follow with an assistant message,
                # we don't need to add the assistant generation prefix
                message_end_idx = tokenizer.apply_chat_template(
                    conversation=messages[: message_idx + 1],
                    tokenize=True,
                    return_tensors="pt",
                    padding=False,
                    truncation=True,
                    max_length=max_seq_length,
                    add_generation_prompt=False,
                ).shape[1]
            # set the label to -100 for the non-assistant part
            labels[:, message_start_idx:message_end_idx] = -100
            if max_seq_length and message_end_idx >= max_seq_length:
                break
    attention_mask = torch.ones_like(input_ids)
    return {
        "input_ids": input_ids.flatten().tolist(),
        "labels": labels.flatten().tolist(),
        "attention_mask": attention_mask.flatten().tolist(),
    }


print(encode_sft_example(input_data[0], verbose=True))

if abs(proportion-1.0)<1e-6:
    tokenized_data = []
    with Pool(args.preprocessing_num_workers) as p:
        pbar = tqdm(input_data, desc=f"tokenizing")
        for tokenized_example in p.imap(encode_sft_example, pbar):
            dump = json.dumps(tokenized_example)
            tokenized_data.append(dump)

    with open(args.output_file, "w") as fw:
        for dump in tokenized_data:
            fw.write(dump + "\n")

else:
    sft_tokenized_data = []
    usft_tokenized_data = []
    sft_data = input_data.select(range(int(len(input_data)*proportion)))
    usft_data = input_data.select(range(int(len(input_data)*proportion),args.end))

    with Pool(args.preprocessing_num_workers) as p:
        pbar_sft = tqdm(sft_data, desc=f"tokenizing supervised data")
        for tokenized_example in p.imap(encode_sft_example, pbar_sft):
            dump = json.dumps(tokenized_example)
            sft_tokenized_data.append(dump)

    with open(args.sft_path, "w") as fw:
        for dump in sft_tokenized_data:
            fw.write(dump + "\n")

    with Pool(args.preprocessing_num_workers) as p:
        pbar = tqdm(usft_data, desc=f"tokenizing unsupervied data")
        for tokenized_example in p.imap(encode_sft_example, pbar):
            dump = json.dumps(tokenized_example)
            usft_tokenized_data.append(dump)

    with open(args.usft_path, "w") as fw:
        for dump in usft_tokenized_data:
            fw.write(dump + "\n")
