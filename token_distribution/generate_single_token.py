import torch
import json
import argparse
import re
from transformers import AutoTokenizer, AutoModelForCausalLM
from entmax import sparsemax
def main():
    parser = argparse.ArgumentParser(description="Collect token distributions when model outputs a number")
    parser.add_argument("--model", type=str, default="gpt2", help="Model name or path")
    parser.add_argument("--num_runs", type=int, default=50, help="Number of generations to try")
    parser.add_argument("--topk", type=int, default=20, help="Top-k tokens to record")
    parser.add_argument("--outfile", type=str, default="digit_distributions.json", help="Output JSON file")
    args = parser.parse_args()

    prompt = """Give me a single number as response.Only output one number and nothing else.Answer:"""

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, device_map="auto")

    all_distributions = []

    for run in range(args.num_runs):
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=10,    # 给足够空间输出数字
                return_dict_in_generate=True,
                output_scores=True,
                do_sample=True,
                temperature=0.6,
                top_p=0.9
            )

        # 找到第一个 "数字 token" 的位置
        seq = outputs.sequences[0]
        start = inputs["input_ids"].size(1)
        gen_tokens = seq[start:]  # 新生成的 token 序列

        target_step = None
        number_text = None

        for step, token_id in enumerate(gen_tokens):
            token_str = tokenizer.decode([token_id], skip_special_tokens=True).strip()
            if re.search(r"\d+", token_str):
                target_step = step
                number_text = token_str
                break

        if target_step is None:
            print(f"Run {run+1}: no digit found, skip")
            continue

        # 提取该数字 token 对应的分布
        logits = outputs.scores[target_step][0]   # shape [vocab_size]
        # probs = torch.softmax(logits, dim=-1)
        probs = sparsemax(logits,dim=-1)
        topk_probs, topk_indices = torch.topk(probs, k=args.topk)
        topk_tokens = tokenizer.convert_ids_to_tokens(topk_indices.tolist())

        dist = {
                f"top{i+1}": {"token": tok, "prob":f'{prob}'}
            for i, (tok, prob) in enumerate(zip(topk_tokens, topk_probs))
        }

        raw_text = tokenizer.decode(seq[start:], skip_special_tokens=True).strip()
        all_distributions.append({"generated": number_text, "raw_text": raw_text, "distribution": dist})

        print(f"Run {run+1}: generated number {number_text}, raw='{raw_text}'")

    # 保存 JSON
    with open(args.outfile, "w", encoding="utf-8") as f:
        json.dump(all_distributions, f, indent=2, ensure_ascii=False)

    print(f"\nSaved {len(all_distributions)} valid digit distributions to {args.outfile}")

if __name__ == "__main__":
    main()

