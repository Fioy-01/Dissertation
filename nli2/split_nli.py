import json, os, math
from collections import defaultdict
from pathlib import Path

def load_json_or_jsonl(p):
    p = Path(p)
    with open(p, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

def save_jsonl(items, path):
    os.makedirs(Path(path).parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for x in items:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")

def norm_item(x):
    return {
        "id": x.get("id", None),
        "premise": x.get("premise", ""),
        "hypothesis": x.get("hypothesis", ""),
        "label": x.get("label", "")
    }

def largest_subset_with_ratio(items, ratio_dict):
    buckets = defaultdict(list)
    for it in items:
        if it["label"] in ratio_dict:
            buckets[it["label"]].append(it)

    alpha = math.inf
    for k, w in ratio_dict.items():
        n_k = len(buckets.get(k, []))
        if w == 0 or n_k == 0:
            alpha = 0
            break
        alpha = min(alpha, n_k // w)
    alpha = int(alpha)

    selected = []
    target_counts = {}
    for k, w in ratio_dict.items():
        take = w * alpha
        target_counts[k] = take
        selected.extend(buckets[k][:take])

    return selected, target_counts

def stratified_take(items, ratio_dict, take_total):
    total_w = sum(ratio_dict.values())
    per_label_take = {k: int(take_total * w / total_w) for k, w in ratio_dict.items()}

    remainder = take_total - sum(per_label_take.values())
    for k in sorted(ratio_dict.keys(), key=lambda x: ratio_dict[x], reverse=True):
        if remainder <= 0:
            break
        per_label_take[k] += 1
        remainder -= 1

    buckets = defaultdict(list)
    for it in items:
        buckets[it["label"]].append(it)

    init = []
    for k, m in per_label_take.items():
        init.extend(buckets[k][:m])

    init_ids = set(id(x) for x in init)
    unlabeled = [x for x in items if id(x) not in init_ids]
    return init, unlabeled, per_label_take

def main():
    train_path = r"mis_gu_train.json"
    val_path   = r"mis_gu_validation.json"
    test_path  = r"mis_gu_test.json"
    outdir     = Path("splits_fixed")

    init_size  = 990
    rounds     = 4
    per_round  = 100

    train = [norm_item(x) for x in load_json_or_jsonl(train_path)]
    val   = [norm_item(x) for x in load_json_or_jsonl(val_path)]
    test  = [norm_item(x) for x in load_json_or_jsonl(test_path)]

    save_jsonl(val, outdir / "fixed_validation.jsonl")
    save_jsonl(test, outdir / "fixed_test.jsonl")

    labels = sorted(set(x["label"] for x in train))
    print("标签集合:", labels)

    scenarios = {
        "ratio_1_1_1":  {"entailment":1, "neutral":1, "contradiction":1},
        "ratio_1_1_8":  {"entailment":1, "neutral":1, "contradiction":8},
        "ratio_1_1_20": {"entailment":1, "neutral":1, "contradiction":20},
    }
    for k in list(scenarios.keys()):
        scenarios[k] = {lbl:w for lbl,w in scenarios[k].items() if lbl in labels}
        if not scenarios[k]:
            del scenarios[k]

    for scen_name, ratio in scenarios.items():
        sub, target_counts = largest_subset_with_ratio(train, ratio)

        if len(sub) < init_size + rounds*per_round:
            print(f"[警告] {scen_name} 可用样本 {len(sub)} 小于总预算 {init_size + rounds*per_round}")

        init, unlabeled, per_label_take = stratified_take(sub, ratio, init_size)

        out_base = outdir / scen_name
        save_jsonl(init, out_base / "init_labeled.jsonl")
        save_jsonl(unlabeled, out_base / "unlabeled_pool.jsonl")

        stats = {
            "scenario": scen_name,
            "ratio": ratio,
            "available_counts": target_counts,
            "init_size": init_size,
            "init_per_label": per_label_take,
            "unlabeled_size": len(unlabeled),
            "rounds": rounds,
            "per_round": per_round,
            "total_budget_after_AL": init_size + rounds*per_round
        }
        with open(out_base / "stats.json", "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)

        print(f"[完成] {scen_name}: 初始={len(init)}, 未标注={len(unlabeled)}, 可用={sum(target_counts.values())}")

if __name__ == "__main__":
    main()
