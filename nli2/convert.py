import pandas as pd

# 读取 JSONL 数据（每行一个 JSON 对象）
df = pd.read_json("misinformation_guard_validation_clean2.json", lines=True)

# 清理 reasoning 前的逗号和空格
df["reasoning"] = df["reasoning"].astype(str).str.lstrip(" ,")

# NLI 标签映射
label_map = {
    3: "entailment",
    1: "neutral",
    2: "neutral",
    0: "contradiction"
}

# 转换成 NLI 格式
df_nli = pd.DataFrame({
    "premise": df["text"],
    "hypothesis": df["reasoning"],
    "label": df["label"].map(label_map)
})

# 保存成 JSONL
df_nli.to_json("mis_guard_validation.json", orient="records", lines=True, force_ascii=False)

print("✅ 已生成 NLI ")
