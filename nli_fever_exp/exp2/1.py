import json
from collections import Counter

label_counter = Counter()
with open("train_fitems.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        item = json.loads(line.strip())
        label = item.get("label")
        label_counter[label] += 1

print("标签统计情况：")
for label, count in label_counter.items():
    print(f"  {label}: {count}")
