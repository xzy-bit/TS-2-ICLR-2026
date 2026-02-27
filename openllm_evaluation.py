import os, re, json, sys
from dataclasses import dataclass, field
from pprint import pprint
from typing import List, Tuple, Dict, Any
from collections import Counter

import torch
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, HfArgumentParser

import vllm
from vllm import SamplingParams

# ----------------------
# Prompt & parsing utils
# ----------------------
GSM_TEMPLATE = """Your task is to answer the question below. 
Give step by step reasoning before you answer, and when you’re ready to answer, 
please use the format "The answer is: ..."

Question: {question}
"""

MC_TEMPLATE = """You will be given a question with multiple-choice answers.
Think step by step briefly, then give ONLY the final choice letter.

Question:
{question}

Choices:
{choices_str}

When you are ready, answer strictly in the format:
The answer is: X
(where X is one of A, B, C, D or E).
"""

LETTER_RE = re.compile(
    r"(?:The\s+answer\s+is\s*[:\-]?\s*([A-E]))\b|(?:\b([A-E])\b(?=[\s\.\)\:,$]))",
    flags=re.IGNORECASE
)

def choices_to_str(choices: List[str]) -> str:
    letters = "ABCDE"
    lines = []
    for i, ch in enumerate(choices):
        lines.append(f"{letters[i]}. {ch}".strip())
    return "\n".join(lines)

def extract_choice_letter(text: str, choices: List[str]) -> str | None:
    # 1) Try to find an explicit letter.
    found = None
    for m in LETTER_RE.finditer(text):
        g = m.group(1) or m.group(2)
        if g:
            found = g.upper()
    if found:
        return found

    # 2) Weak fallback: if it literally quotes one choice string, map it.
    text_low = text.lower()
    for i, ch in enumerate(choices):
        if ch.lower() in text_low:
            return "ABCDE"[i]
    return None

def extract_gsm_number(text: str) -> str | None:
    text_new = text.replace("the answer is", "The answer is")
    parts = text_new.split("The answer is")
    if len(parts) > 1:
        extract_ans = parts[-1].strip()
        m = re.search(r"[-+]?\d*[\.,/]?\d+", extract_ans)
        if m:
            return m.group(0).replace(",", "")
    return None

def majority_and_best_of_n(reference, candidates, depths=(1,4,8,16,32)):
    maj, bon = [], []
    for d in depths:
        if d > len(candidates):
            break
        window = [c for c in candidates[:d] if c is not None]
        # Majority vote
        if window:
            counts = Counter(window)
            majority_choice = counts.most_common(1)[0][0]
        else:
            majority_choice = None
        maj.append(1 if majority_choice == reference else 0)
        # Best-of-n
        bon.append(1 if reference in window else 0)
    return maj, bon

# ---------------
# Dataset loaders
# ---------------
def load_gsm8k(split="test"):
    ds = load_dataset("gsm8k", "main")[split]
    items = []
    for r in ds:
        q = r["question"]
        ans = r["answer"]
        m = re.search(r"####\s*([0-9\.\-]+)", ans)
        if not m:
            continue
        gt_number = m.group(1).strip()
        items.append((q, gt_number))   # 二元组
    return items

def load_arc(split="validation"):
    ds = load_dataset("ai2_arc", "ARC-Challenge")[split]
    items = []
    for r in ds:
        q = r["question"]
        ch = r["choices"]["text"]
        ans_label = r["answerKey"].strip().upper()
        items.append((q, ch, ans_label))
    return items

def load_hellaswag(split="validation"):
    ds = load_dataset("hellaswag")[split]
    items = []
    for r in ds:
        q = (r["ctx"] + " " + r["ctx_a"]).strip()
        ch = r["endings"]
        ans_idx = int(r["label"])
        ans_letter = "ABCD"[ans_idx]
        items.append((q, ch, ans_letter))
    return items

def load_winogrande(split="validation"):
    ds = load_dataset("winogrande", "winogrande_debiased")[split]
    items = []
    for r in ds:
        q = r["sentence"].replace("_", "_____")
        ch = [r["option1"], r["option2"]]
        ans_idx = 0 if r["answer"].strip() == "1" else 1
        ans_letter = "AB"[ans_idx]
        items.append((q, ch, ans_letter))
    return items

def load_mmlu():
    def _rows_to_triples(rows):
        items = []
        for r in rows:
            if "choices" in r and isinstance(r["choices"], (list, tuple)):
                choices = list(r["choices"])
            else:
                choices = [r.get("A"), r.get("B"), r.get("C"), r.get("D")]
            ans = r.get("answer")
            if isinstance(ans, str):
                gt_letter = ans.strip().upper()
            else:
                gt_letter = "ABCD"[int(ans)]
            items.append((r["question"], choices, gt_letter))
        return items
    try:
        ds = load_dataset("lukaemon/mmlu")
        split = "test" if "test" in ds else ("validation" if "validation" in ds else "dev")
        return _rows_to_triples(ds[split])
    except Exception as e:
        print(f"[warn] lukaemon/mmlu failed ({e}). Falling back to cais/mmlu.")
        ds = load_dataset("cais/mmlu", "all")
        return _rows_to_triples(ds["test"])

def load_truthfulqa_mc1(split="validation"):
    ds = load_dataset("truthful_qa", "multiple_choice")[split]
    items = []
    for r in ds:
        q = r["question"]
        tgt = r.get("mc1_targets", None) or r.get("mc1_targets_scores", None)
        if tgt and "choices" in tgt:
            ch = tgt["choices"]
            labels = tgt.get("labels", None)
            if labels:
                idxs = [i for i, v in enumerate(labels) if int(v) == 1]
                if idxs:
                    ans_letter = "ABCDE"[idxs[0]]
                else:
                    correct = r.get("mc1_correct", None)
                    if correct and correct in ch:
                        ans_letter = "ABCDE"[ch.index(correct)]
                    else:
                        continue
            else:
                correct = r.get("mc1_correct", None)
                if correct and correct in ch:
                    ans_letter = "ABCDE"[ch.index(correct)]
                else:
                    continue
            ch = ch[:5]
            items.append((q, ch, ans_letter))
    return items

TASK_LOADERS = {
    "arc": load_arc,
    "hellaswag": load_hellaswag,
    "winogrande": load_winogrande,
    "mmlu": load_mmlu,
    "truthfulqa": load_truthfulqa_mc1,
    "gsm8k": load_gsm8k,
}

# -----------
# CLI config
# -----------
@dataclass
class Arguments:
    task: str = field(default="arc")
    model_name_or_path: str = field(default="")
    tokenizer_name_or_path: str = field(default="")
    dtype: str = field(default="bf16", metadata={"choices": ["fp16", "bf16"]})
    temperature: float = 0.6
    top_k: int = -1
    top_p: float = 0.9
    n: int = 32
    max_new_tokens: int = 256
    batch_size: int = 16
    use_vllm: bool = True
    vllm_gpu_memory_utilization: float = 0.9
    seed: int = 42
    remove_old: bool = False
    save_path: str = field(default="")
    summary_path: str = field(default="")
    depths_csv: str = field(default="1,4,8,16,32")

def main():
    parser = HfArgumentParser((Arguments,))
    (args,) = parser.parse_args_into_dataclasses()
    pprint(args.__dict__)

    if args.remove_old:
        for p in [args.save_path, args.summary_path]:
            if p and os.path.exists(p):
                os.remove(p)

    task = args.task.lower()
    if task not in TASK_LOADERS:
        raise ValueError(f"Unknown task '{task}'. Choose from {list(TASK_LOADERS.keys())}")
    items = TASK_LOADERS[task]()
    print(f"[info] Loaded {len(items)} items for task: {task}")

    device = torch.device("cuda")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name_or_path)
    if "llama-3" in tokenizer.name_or_path.lower():
        tokenizer.pad_token = tokenizer.decode(len(tokenizer) - 1)
        tokenizer.pad_token_id = len(tokenizer) - 1
    elif "llama-2" in tokenizer.name_or_path.lower():
        tokenizer.pad_token = tokenizer.unk_token
        tokenizer.pad_token_id = tokenizer.unk_token_id

    torch_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}[args.dtype]

    if not args.use_vllm:
        raise NotImplementedError("This script currently supports vLLM only.")

    llm = vllm.LLM(
        model=args.model_name_or_path,
        tokenizer=args.tokenizer_name_or_path,
        tensor_parallel_size=torch.cuda.device_count(),
        dtype=torch_dtype,
        gpu_memory_utilization=args.vllm_gpu_memory_utilization,
        seed=args.seed,
    )

    depths = tuple(int(x) for x in args.depths_csv.split(",") if x.strip())
    save_rows: List[Dict[str, Any]] = []
    maj_all = np.zeros((len(items), len(depths)), dtype=int)
    bon_all = np.zeros((len(items), len(depths)), dtype=int)

    for i in range(0, len(items), args.batch_size):
        batch = items[i : i + args.batch_size]
        qs, choices_batch, gts = [], [], []

        if task == "gsm8k":
            for q, gold_number in batch:
                qs.append(q)
                gts.append(gold_number)
                choices_batch.append(None)
        else:
            for q, ch, gt_letter in batch:
                qs.append(q)
                choices_batch.append(ch)
                gts.append(gt_letter)

        conv = []
        for q, ch in zip(qs, choices_batch):
            if task == "gsm8k":
                prompt = GSM_TEMPLATE.format(question=q)
            else:
                prompt = MC_TEMPLATE.format(question=q, choices_str=choices_to_str(ch))
            conv.append([{"role": "user", "content": prompt}])

        tokenizer.padding_side = "left"
        prompt_token = tokenizer.apply_chat_template(
            conv,
            padding="longest",
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        ).to(device)

        prompt_token_ids = [
            prompt_token.input_ids[j, prompt_token.attention_mask[j].bool()].tolist()
            for j in range(len(conv))
        ]

        eff_top_k = args.top_k if (args.top_k is not None and args.top_k > 0) else -1
        sampling = SamplingParams(
            n=args.n,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=eff_top_k,
            max_tokens=args.max_new_tokens,
        )

        with torch.no_grad():
            outputs = llm.generate(prompt_token_ids,
                                   sampling_params=sampling)

        for j, out in enumerate(outputs):
            all_texts = [out.outputs[k].text for k in range(len(out.outputs))]
            if task == "gsm8k":
                pred_numbers = [extract_gsm_number(t) for t in all_texts]
                gold_number = gts[j]
                maj, bon = majority_and_best_of_n(gold_number, pred_numbers, depths=depths)
                save_rows.append({
                    "id": i + j,
                    "prompt": conv[j],
                    "gold": gold_number,
                    "responses": all_texts,
                    "parsed_numbers": pred_numbers,
                    "majority_eval": maj,
                    "best_of_n_eval": bon,
                })
            else:
                letters = [extract_choice_letter(t, choices_batch[j]) for t in all_texts]
                maj, bon = majority_and_best_of_n(gts[j], letters, depths=depths)
                save_rows.append({
                    "id": i + j,
                    "prompt": conv[j],
                    "choices": choices_batch[j],
                    "gold": gts[j],
                    "responses": all_texts,
                    "parsed_letters": letters,
                    "majority_eval": maj,
                    "best_of_n_eval": bon,
                })

            maj_all[i + j, :len(maj)] = np.array(maj, dtype=int)
            bon_all[i + j, :len(bon)] = np.array(bon, dtype=int)

        if (i // args.batch_size) % 8 == 0 and args.save_path:
            with open(args.save_path, "w", encoding="utf-8") as f:
                json.dump(save_rows, f, indent=2)

    if args.save_path:
        with open(args.save_path, "w", encoding="utf-8") as f:
            json.dump(save_rows, f, indent=2)

    maj_acc = np.mean(maj_all, axis=0).tolist()
    bon_acc = np.mean(bon_all, axis=0).tolist()
    summary = {
        "task": task,
        "depths": list(depths),
        "majority_vote_accuracy": maj_acc,
        "best_of_n_accuracy": bon_acc,
        "n_samples": args.n,
        "count": len(items),
    }
    pprint(summary)
    if args.summary_path:
        with open(args.summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

if __name__ == "__main__":
    main()
