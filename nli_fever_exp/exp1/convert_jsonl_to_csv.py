import json
import pandas as pd

# 标签映射字典
label2id = {"SUPPORTS": 0, "REFUTES": 1, "NOT ENOUGH INFO": 2}

def convert_jsonl_to_csv(jsonl_file, csv_file):
    data = []
    with open(jsonl_file, "r", encoding="utf-8") as f:
        for line in f:
            example = json.loads(line)
            text = example["query"] + " " + example["context"]
            label = label2id[example["label"]]
            data.append({"text": text, "label": label})
    df = pd.DataFrame(data)
    df.to_csv(csv_file, index=False)
    print(f"✅ Converted {jsonl_file} → {csv_file}")

# 示例转换
convert_jsonl_to_csv("init_labeled_set.jsonl", "init_labeled_set.csv")
convert_jsonl_to_csv("unlabeled_pool.jsonl", "unlabeled_pool.csv")
convert_jsonl_to_csv("validation_set.jsonl", "validation_set.csv")
convert_jsonl_to_csv("train_fitems.jsonl", "train_fitems.csv")
