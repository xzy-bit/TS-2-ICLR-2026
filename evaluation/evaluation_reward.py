import os
from dataclasses import dataclass, field
from pprint import pprint
import json
from types import MethodType
from tqdm import tqdm
import numpy as np

import torch
from transformers import AutoTokenizer, pipeline
from transformers import AutoModel, AutoModelForSequenceClassification, HfArgumentParser


@dataclass
class Arguments:
    model_name_or_path: str = field(default="sfairXC/FsfairX-LLaMA3-RM-v0.1")
    tokenizer_path: str = field(default=None)

    detokenizer_path: str = field(default=None)

    data_path: str = field(default=None)
    batch_size: int = field(default=1)
    max_size: int = field(default=None)

    save_path: str = field(default=None)


def forward_value_fn(
    self,
    input_ids=None,
    attention_mask=None,
    past_key_values=None,
    position_ids=None,
    inputs_embeds=None,
    return_value_only=False,
    prompt_length=0,
    use_cache=False,
    **kwargs,
):
    transformer_outputs = self.model(
        input_ids,
        past_key_values=past_key_values,
        attention_mask=attention_mask,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        **kwargs,
    )
    hidden_states = transformer_outputs[0]
    values = self.score(hidden_states).squeeze(-1)
    if return_value_only:
        return values
    else:
        if attention_mask is None:
            chosen_end_scores = values[:, -1]
        else:
            last_index = attention_mask.cumsum(dim=1).argmax(dim=1)
            chosen_end_scores = values.gather(1, last_index.unsqueeze(1)).squeeze(1)
    return {
        "values": values,
        "chosen_end_scores": chosen_end_scores,
    }


def calculation_best_of_n(data):
    print("Calculating best of n reward ....")
    best_n = np.zeros([len(data), 6])  # 1, 2, 4, 8, 16
    mean_n = np.zeros([len(data), 6])  # 1, 2, 4, 8, 16
    for i in tqdm(range(len(data))):
        rewards = data[i]["reward"]
        best_n[i][0] = rewards[0]
        best_n[i][1] = max(rewards[:2])
        best_n[i][2] = max(rewards[:4])
        best_n[i][3] = max(rewards[:8])
        best_n[i][4] = max(rewards[:16])
        best_n[i][5] = max(rewards[:32])

        mean_n[i][0] = rewards[0]
        mean_n[i][1] = np.mean(rewards[:2])
        mean_n[i][2] = np.mean(rewards[:4])
        mean_n[i][3] = np.mean(rewards[:8])
        mean_n[i][4] = np.mean(rewards[:16])
        mean_n[i][5] = np.mean(rewards[:32])
    best_n = np.mean(best_n, axis=0)
    print("Best of n: {}".format(np.round(best_n, 2)))
    mean_n = np.mean(mean_n, axis=0)
    print("Mean of n: {}".format(np.round(mean_n, 2)))
    return best_n, mean_n

def calculate_winrate(data):
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
    print("Execllent rate of n:")
    for item in np.round(best_n,2):
        print(item,end='\t')
    print('\n')
    return best_n


def main():
    parser = HfArgumentParser((Arguments,))
    (args,) = parser.parse_args_into_dataclasses()
    pprint(args.__dict__)
    assert args.data_path is not None
    assert args.save_path is not None

    device = torch.device("cuda")

    model_class = AutoModelForSequenceClassification
    flash_attn = False
    model = model_class.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2" if flash_attn else "eager",
        trust_remote_code=True,
    )
    # model.forward_value = forward_value_fn
    model.forward_value = MethodType(forward_value_fn, model)
    model.eval()
    model.to(device)

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_path or args.model_name_or_path
    )
    tokenizer.padding_side = "right"
    if args.detokenizer_path is not None:
        detokenizer = AutoTokenizer.from_pretrained(args.detokenizer_path)
    else:
        detokenizer = None

    response_data = json.load(open(args.data_path, "r"))

    if args.max_size:
        response_data = response_data[: args.max_size]
    if os.path.exists(args.save_path):
        response_data = json.load(open(args.save_path, "r"))
        calculation_best_of_n(response_data)
        calculate_winrate(response_data)
        return
    for start in tqdm(range(0, len(response_data), args.batch_size)):
        end = start + args.batch_size
        prompts = []
        answers = []
        for x in response_data[start:end]:
            if detokenizer:
                prompt_str = (
                    detokenizer.decode(
                        detokenizer.encode(x["prompt"]), skip_special_tokens=True
                    )
                    .replace("user\n\n", "")
                    .replace("assistant\n\n", "")
                )
            else:
                if "prompt" in x:
                    prompt_str = x["prompt"]
                elif "instruction" in x:
                    prompt_str = x["instruction"]
                else:
                    raise ValueError(x)
            if "answer" in x:
                for ans in x["answer"]:
                    if detokenizer:
                        ans_str = detokenizer.decode(
                            detokenizer.encode(ans), skip_special_tokens=True
                        )
                    else:
                        ans_str = ans
                    prompts.append(prompt_str)
                    answers.append(ans_str)
            elif "output" in x:
                ans_str = x["output"]
                prompts.append(prompt_str)
                answers.append(ans_str)
            else:
                raise ValueError(x)

        chat = []
        for i in range(len(prompts)):
            chat.append(
                [
                    {"role": "user", "content": prompts[i]},
                    {"role": "assistant", "content": answers[i]},
                ]
            )
        inputs = tokenizer.apply_chat_template(
            chat,
            padding="longest",
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            if "FsfairX-LLaMA3-RM-v0.1" in args.model_name_or_path:
                outputs = model.forward_value(**inputs)["chosen_end_scores"]
            else:
                outputs = model(**inputs, use_cahe=False)

        c_start = 0
        for x in response_data[start:end]:
            if "answer" in x:
                x["reward"] = outputs[c_start : c_start + len(x["answer"])].tolist()
                c_start += len(x["answer"])
            elif "output" in x:
                x["reward"] = outputs[c_start].tolist()
                c_start += 1
            else:
                raise ValueError(x)

        print(chat[0])
        print(outputs[0])

    if "answer" in x:
        calculation_best_of_n(response_data)
        calculate_winrate(response_data)
    json.dump(response_data, open(args.save_path,"w"), separators=(",", ":"))
    # json.dump(response_data, open(args.save_path, "w"), indent=2)
    print("saving result to {}".format(args.save_path))


if __name__ == "__main__":
    main()
