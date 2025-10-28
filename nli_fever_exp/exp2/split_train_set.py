import json
import random
from sklearn.model_selection import train_test_split

# 设定随机种子确保可复现
random.seed(42)

# 读取数据
with open("train_fitems.jsonl", "r", encoding="utf-8") as f:
    data = [json.loads(line.strip()) for line in f if line.strip()]

print(f"原始样本总数: {len(data)}")

# stratify使用label，保证类别平衡（前提是每条都有label）
labels = [item['label'] for item in data]

# 先划出验证集 15%
train_pool, val_set = train_test_split(data, test_size=0.15, stratify=labels, random_state=42)

# 从剩下的 pool 中划出 5% 作为初始标注集
pool_labels = [item['label'] for item in train_pool]
init_labeled_set, unlabeled_pool = train_test_split(train_pool, test_size=0.95, stratify=pool_labels, random_state=42)

# 输出结果统计
print(f"初始标注集数量: {len(init_labeled_set)}")
print(f"未标注池数量: {len(unlabeled_pool)}")
print(f"验证集数量: {len(val_set)}")

# 保存文件函数
def save_jsonl(filename, dataset):
    with open(filename, "w", encoding="utf-8") as f:
        for item in dataset:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

save_jsonl("init_labeled_set.jsonl", init_labeled_set)
save_jsonl("unlabeled_pool.jsonl", unlabeled_pool)
save_jsonl("validation_set.jsonl", val_set)

print("✅ 划分完成，3个文件已保存。")
