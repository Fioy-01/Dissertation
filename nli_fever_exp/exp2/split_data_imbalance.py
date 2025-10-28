import json
import random
from sklearn.model_selection import train_test_split
from collections import defaultdict

random.seed(42)

# 读取数据
with open("train_fitems.jsonl", "r", encoding="utf-8") as f:
    data = [json.loads(line.strip()) for line in f if line.strip()]

print(f"原始样本总数: {len(data)}")

# 分类汇总
label_groups = defaultdict(list)
for item in data:
    label_groups[item["label"]].append(item)

# 构建初始标注集（1:1:8 比例）
init_labeled_set = []
init_labeled_set += random.sample(label_groups["SUPPORTS"], 100)
init_labeled_set += random.sample(label_groups["REFUTES"], 100)
init_labeled_set += random.sample(label_groups["NOT ENOUGH INFO"], 800)

# 将初始标注集从数据中移除，剩下的是训练池
init_ids = set(id(item) for item in init_labeled_set)
unlabeled_pool = [item for item in data if id(item) not in init_ids]

# 从训练池中划分出 15% 作为验证集
pool_labels = [item["label"] for item in unlabeled_pool]
train_pool, val_set = train_test_split(unlabeled_pool, test_size=0.15, stratify=pool_labels, random_state=42)

# 保存数据函数
def save_jsonl(filename, dataset):
    with open(filename, "w", encoding="utf-8") as f:
        for item in dataset:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

# ✅ 修正后的保存步骤
save_jsonl("init_labeled_set_imb_118.jsonl", init_labeled_set)
save_jsonl("unlabeled_pool_imb_118.jsonl", train_pool)
save_jsonl("validation_set_imb_118.jsonl", val_set)

print("✅ 不平衡数据划分完成：1:1:8，三个文件已保存。")
