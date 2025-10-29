from datasets import load_dataset

# 1. 下载数据集
dataset = load_dataset("Intel/misinformation-guard")

# 2. 保存为 JSON 文件
dataset["train"].to_json("misinformation_guard_train.json", orient="records", lines=True)
dataset["test"].to_json("misinformation_guard_test.json", orient="records", lines=True)
dataset["validation"].to_json("misinformation_guard_validation.json", orient="records", lines=True)

print("✅ 数据已保存为 JSON 文件")


