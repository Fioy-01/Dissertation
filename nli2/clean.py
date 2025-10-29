import pandas as pd
import re

# 读取现有 NLI JSONL
df = pd.read_json("mis_guard_validation.json", lines=True)

def remove_leading_phrase(text):
    if not isinstance(text, str):
        return text
    # 删除开头的 "This statement is true and factual information."
    text = re.sub(r"^This statement is true and factual information\. ?", "", text, flags=re.IGNORECASE)
    return text.strip()

# 应用到 hypothesis
df["hypothesis"] = df["hypothesis"].apply(remove_leading_phrase)

# 保存清理后的数据
df.to_json("mis_gu_validation.json", orient="records", lines=True, force_ascii=False)

print("已生成清理开头套话的 NLI 数据集")
