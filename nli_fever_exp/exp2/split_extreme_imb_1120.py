# split_extreme_imb_1120.py
import json, random
from collections import defaultdict
from sklearn.model_selection import train_test_split

random.seed(42)

with open("train_fitems.jsonl", "r", encoding="utf-8") as f:
    data = [json.loads(line) for line in f]

label_groups = defaultdict(list)
for item in data:
    label_groups[item["label"]].append(item)

init_labeled_set = []
init_labeled_set += random.sample(label_groups["SUPPORTS"], 100)
init_labeled_set += random.sample(label_groups["REFUTES"], 100)
init_labeled_set += random.sample(label_groups["NOT ENOUGH INFO"], 2000)

init_ids = set(id(i) for i in init_labeled_set)
unlabeled_pool = [x for x in data if id(x) not in init_ids]
train_pool, val_set = train_test_split(unlabeled_pool, test_size=0.15, stratify=[d["label"] for d in unlabeled_pool], random_state=42)

def save_jsonl(name, items):
    with open(name, "w", encoding="utf-8") as f:
        for x in items:
            f.write(json.dumps(x) + "\n")

save_jsonl("init_labeled_set_extreme_imb_1120.jsonl", init_labeled_set)
save_jsonl("unlabeled_pool_extreme_imb_1120.jsonl", train_pool)
save_jsonl("validation_set_extreme_imb_1120.jsonl", val_set)
