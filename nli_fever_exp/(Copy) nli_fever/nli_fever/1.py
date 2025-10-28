import json

# === 配置 ===
input_file = "dev_fitems.jsonl"     # 原始 FEVER 格式文件路径
output_file = "dev_f.jsonl"      # 转换后的 NLI 格式文件路径

# FEVER -> NLI 标签映射
label_map = {
    "SUPPORTS": "entailment",
    "REFUTES": "contradiction",
    "NOT ENOUGH INFO": "neutral"
}

def convert_fever_to_nli(input_path, output_path):
    new_data = []
    with open(input_path, "r", encoding="utf-8") as fin:
        for idx, line in enumerate(fin, start=1):
            item = json.loads(line.strip())

            premise = item.get("context", "").strip()
            hypothesis = item.get("query", "").strip()
            fever_label = item.get("label", "").strip()

            # 转换标签
            nli_label = label_map.get(fever_label, "neutral")  # 默认 neutral 防止出错

            # 生成新格式
            new_item = {
                "id": idx,
                "premise": premise,
                "hypothesis": hypothesis,
                "label": nli_label
            }
            new_data.append(new_item)

    # 保存为 JSONL
    with open(output_path, "w", encoding="utf-8") as fout:
        for item in new_data:
            fout.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"转换完成！共 {len(new_data)} 条样本，已保存到 {output_path}")

if __name__ == "__main__":
    convert_fever_to_nli(input_file, output_file)
