import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="gpt2")
    parser.add_argument("--prompt", type=str, default="Give me a single-digit number")
    parser.add_argument("--topk", type=int, default=300)
    parser.add_argument("--output_file", type=str, default="next_token_probs.json")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(args.model_name)
    model.eval()

    inputs = tokenizer(args.prompt, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits[0, -1, :]

    probs = torch.softmax(logits, dim=-1)
    probs_topk, indices_topk = torch.topk(probs, args.topk)

    probs_dict = {
        tokenizer.decode([idx.item()]): float(prob.item())
        for idx, prob in zip(indices_topk, probs_topk)
    }

    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(probs_dict, f, indent=2)

    print(f"Saved top-{args.topk} token probabilities to {args.output_file}")

if __name__ == "__main__":
    main()

