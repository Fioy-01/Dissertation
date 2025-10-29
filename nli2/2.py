import os
import json

# 根目录
root_dir = r"D:\PROJECT\Dissertation\nli2\data"

def update_jsonl_ids(file_path):
    updated_lines = []
    changed = False

    with open(file_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f, start=1):
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                print(f"[跳过无效行] {file_path} 第 {idx} 行")
                continue

            if data.get("id") is None:  # 如果id是null
                data["id"] = idx  # 用行号当id（也可以换成uuid等）
                changed = True
            updated_lines.append(json.dumps(data, ensure_ascii=False))

    if changed:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write("\n".join(updated_lines))
        print(f"[更新完成] {file_path}")
    else:
        print(f"[无更改] {file_path}")

# 遍历文件夹
for folder, _, files in os.walk(root_dir):
    for file in files:
        if file.endswith(".jsonl"):
            update_jsonl_ids(os.path.join(folder, file))
