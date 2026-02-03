import os
import json
from collections import defaultdict

base_dir = "evals_openllm"

# 按 task 分组存储
task_groups = defaultdict(list)

# 收集数据
for model in sorted(os.listdir(base_dir)):
    model_path = os.path.join(base_dir, model)
    if not os.path.isdir(model_path):
        continue
    for task in sorted(os.listdir(model_path)):
        task_path = os.path.join(model_path, task, f"{task}_voting_summary.json")
        if not os.path.exists(task_path):
            continue
        with open(task_path) as f:
            data = json.load(f)

        best = data.get("best_of_n_accuracy", [])
        majority = data.get("majority_vote_accuracy", [])

        if len(best) < 5 or len(majority) < 5:
            continue

        row = [
            model,
            task,
            best[0]*100, best[1]*100, best[2]*100, best[3]*100, best[4]*100,
            majority[0]*100, majority[1]*100, majority[2]*100, majority[3]*100, majority[4]*100
        ]
        task_groups[task].append(row)


header_fmt = "{:<20}{:<20}" + "{:<8}"*10
print(header_fmt.format("model","task","best@1","best@2","best@4","best@8","best@16","best@32",
                        "maj@1","maj@2","maj@4","maj@8","maj@16"))

for task, rows in sorted(task_groups.items()):
    max_vals = []
    for col in range(2, len(rows[0])):
        max_vals.append(max(row[col] for row in rows))

    for row in rows:
        formatted = []
        for i, val in enumerate(row):
            if i < 2:  # model/task
                formatted.append(f"{val:<20}")
            else:
                is_max = abs(val - max_vals[i-2]) < 1e-9
                txt = f"{val:.2f}\t"
                if is_max:
                    txt = f"\033[1m{txt}\033[0m"
                formatted.append(txt)
        print("".join(formatted))
    print()

