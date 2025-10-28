import json
from collections import Counter

# 你的训练集文件路径
train_file = "train_f.jsonl"

# 读取并统计标签
label_counter = Counter()
total = 0

with open(train_file, "r", encoding="utf-8") as f:
    for line in f:
        item = json.loads(line)
        label = item["label"]
        label_counter[label] += 1
        total += 1

# 输出比例
for label, count in label_counter.items():
    print(f"{label}: {count} ({count/total:.2%})")

print(f"总样本数: {total}")
