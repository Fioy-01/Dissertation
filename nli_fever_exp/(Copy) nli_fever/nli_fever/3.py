# -*- coding: utf-8 -*-
import os, json, random
from collections import defaultdict, Counter
from math import floor

random.seed(42)

# 如需 FEVER 原标注到 NLI 三类的映射，取消注释
# LABELS_FEVER_MAP = {
#     "SUPPORTS": "entailment",
#     "REFUTES": "contradiction",
#     "NOT ENOUGH INFO": "neutral",
#     "NOT_ENOUGH_INFO": "neutral",
# }

LABELS = ["entailment", "neutral", "contradiction"]

def load_jsonl(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ex = json.loads(line)
            # if ex["label"] in LABELS_FEVER_MAP:
            #     ex["label"] = LABELS_FEVER_MAP[ex["label"]]
            if ex.get("label") not in LABELS:
                # 过滤异常标签
                continue
            data.append(ex)
    return data

def save_jsonl(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for x in data:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")

def key_of(ex):
    # 用 (premise, hypothesis, label) 作为样本唯一键；如需更严格可加入 id
    return (ex.get("premise","").strip(), ex.get("hypothesis","").strip(), ex.get("label",""))

def dedup_keep_first(rows):
    seen, out = set(), []
    for s in rows:
        k = key_of(s)
        if k not in seen:
            seen.add(k)
            out.append(s)
    return out

def stratified_split_fixed_counts(rows, take_per_label, seed=42):
    """按标签固定数量抽样（take_per_label 是 {label: k}）。返回 (picked, rest)。"""
    random.seed(seed)
    by_lab = defaultdict(list)
    for s in rows:
        by_lab[s["label"]].append(s)
    for lab, need in take_per_label.items():
        if len(by_lab[lab]) < need:
            raise ValueError(f"[stratified_split_fixed_counts] {lab} 不足: 需 {need}, 有 {len(by_lab[lab])}")
        random.shuffle(by_lab[lab])
    picked, rest = [], []
    for lab, need in take_per_label.items():
        picked.extend(by_lab[lab][:need])
        rest.extend(by_lab[lab][need:])
    # 打乱剩余集
    random.shuffle(rest)
    return picked, rest

def stratified_split_by_ratio(rows, val_frac=0.1, test_frac=0.1, seed=42):
    """
    先从 rows 分层切出 val/test（占比），返回 (val, test, train_rest)。
    占比是对每个标签分别计算，尽量向下取整，保证不超额。
    """
    assert 0.0 <= val_frac < 1.0 and 0.0 <= test_frac < 1.0 and (val_frac + test_frac) < 1.0
    random.seed(seed)
    by_lab = defaultdict(list)
    for s in rows:
        by_lab[s["label"]].append(s)
    for lab in LABELS:
        random.shuffle(by_lab[lab])
    val, test, train_rest = [], [], []
    for lab in LABELS:
        n = len(by_lab[lab])
        nv = floor(n * val_frac)
        nt = floor(n * test_frac)
        # 切片
        val.extend(by_lab[lab][:nv])
        test.extend(by_lab[lab][nv:nv+nt])
        train_rest.extend(by_lab[lab][nv+nt:])
    random.shuffle(val); random.shuffle(test); random.shuffle(train_rest)
    return val, test, train_rest

def count_print(data, title):
    c = Counter([d["label"] for d in data])
    n = len(data)
    print(f"== {title} (N={n}) ==")
    for lab in LABELS:
        pct = (c[lab]/n*100.0) if n else 0.0
        print(f"{lab:14s}: {c[lab]:5d}  ({pct:5.2f}%)")
    print()

def assert_disjoint(A, B, A_name="A", B_name="B"):
    setA = set(map(key_of, A))
    setB = set(map(key_of, B))
    inter = setA & setB
    if inter:
        raise AssertionError(f"[泄漏] {A_name} 与 {B_name} 有交集 {len(inter)} 条样本")

def main():
    # === 1) 输入与参数 ===
    input_train = "train_f.jsonl"   # 你的 FEVER 训练集（已转 NLI 三字段）
    out_root = "fever_splits_2000"
    val_frac = 0.1                  # 验证占比（可改为固定数量：见下可选段）
    test_frac = 0.1                 # 测试占比
    seed = 42

    os.makedirs(out_root, exist_ok=True)

    # === 2) 读入与去重 ===
    all_data = dedup_keep_first(load_jsonl(input_train))
    count_print(all_data, "Full train (dedup)")

    # === 3) 先切 dev/test，再做 AL 训练全集 ===
    val_rows, test_rows, train_universe = stratified_split_by_ratio(
        all_data, val_frac=val_frac, test_frac=test_frac, seed=seed
    )
    count_print(val_rows,  "DEV(val)  reserved")
    count_print(test_rows, "TEST      reserved")
    count_print(train_universe, "Train universe for AL")

    # 互斥性检查
    assert_disjoint(val_rows, test_rows, "val", "test")
    assert_disjoint(val_rows, train_universe, "val", "train_universe")
    assert_disjoint(test_rows, train_universe, "test", "train_universe")

    # 保存公共的 dev/test（所有比例共享）
    common_dir = os.path.join(out_root, "common_eval")
    save_jsonl(os.path.join(common_dir, "val.jsonl"),  val_rows)
    save_jsonl(os.path.join(common_dir, "test.jsonl"), test_rows)

    # === 4) 在“训练全集”上为各比例方案生成 init+pool ===
    plans = {
        "ratio1_real":   {"entailment":1187, "contradiction":471, "neutral":342},  # 2000 总量
        "ratio2_1_1_8":  {"entailment":200,  "neutral":200,      "contradiction":1600},
        "ratio3_1_1_40": {"entailment":48,   "neutral":48,       "contradiction":1904},
    }

    for name, tgt in plans.items():
        # 注意：为了确保各比例完全独立，这里对 train_universe 做一个副本再抽样
        # 如果你希望三个比例共享同一个 pool，可改为在第一次抽完后把 pool 保存为公共版本，其余直接复制。
        picked, pool_rest = stratified_split_fixed_counts(train_universe, tgt, seed=seed)
        out_dir = os.path.join(out_root, name)
        save_jsonl(os.path.join(out_dir, "init_labeled_2000.jsonl"), picked)
        save_jsonl(os.path.join(out_dir, "unlabeled_pool.jsonl"),   pool_rest)
        count_print(picked, f"{name}/init_labeled_2000")
        # 再做一次互斥性检查
        assert_disjoint(picked, val_rows,  f"{name}/init", "val")
        assert_disjoint(picked, test_rows, f"{name}/init", "test")
        assert_disjoint(pool_rest, val_rows,  f"{name}/pool", "val")
        assert_disjoint(pool_rest, test_rows, f"{name}/pool", "test")

    print("完成：已在", out_root, "下写出 common_eval/val.jsonl, test.jsonl 及各比例的 init/pool。")

if __name__ == "__main__":
    main()




