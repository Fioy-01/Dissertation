import os, csv, json, argparse, math, random
from typing import List, Tuple, Dict

from transformers.utils import logging as hf_logging
hf_logging.set_verbosity_error()

from datasets import load_jsonl, save_jsonl, split_by_indices, NliDataset
from models import Classifier
from train_utils import set_seed, build_loader, train_with_early_stop
from sampling import select, entropy_select
from llm_qwen import QwenJudge

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def parse_ratio(ratio_str: str) -> List[int]:
    parts = ratio_str.split(":")
    nums = [int(x) for x in parts]
    if len(nums) != 3 or sum(n <= 0 for n in nums):
        raise ValueError(f"Bad ratio: {ratio_str}")
    return nums

def compute_quota(b: int, ratio: List[int]) -> List[int]:
    s = sum(ratio)
    raw = [b * r / s for r in ratio]
    base = [int(math.floor(x)) for x in raw]
    rem = b - sum(base)
    # distribute remaining by largest fractional parts
    frac = sorted([(raw[i]-base[i], i) for i in range(3)], reverse=True)
    for k in range(rem):
        base[frac[k][1]] += 1
    return base  # [E,N,C], sum==b

def predict_probs_and_labels(model, tokenizer, data_rows, batch_size=32, max_len=256):
    ds = NliDataset(data_rows, tokenizer, max_len=max_len)
    loader = build_loader(ds, batch_size=batch_size, shuffle=False,
                          tokenizer=tokenizer, max_len=max_len)
    out = model.predict(loader)  # expect dict with 'probs' (N,3)
    probs = out["probs"]
    preds = probs.argmax(axis=1).tolist()
    return probs, preds

def entropy_of_probs(p):
    import numpy as np
    eps = 1e-12
    return float(-np.sum(p * np.log(p + eps)))

def s2_select_pool_random(pool, preds, quota, seed=0):
    random.seed(seed)
    idxs = [i for i in range(len(pool))]
    buckets = {0: [], 1: [], 2: []}
    for i, y in zip(idxs, preds):
        buckets[y].append(i)
    take = []
    for y, q in enumerate(quota):
        cand = buckets[y]
        if len(cand) <= q:
            take.extend(cand)
        else:
            take.extend(random.sample(cand, q))
    return take

def s2_select_pool_entropy(pool, probs, preds, quota):
    # rank by entropy within each predicted label
    scored = []
    for i, (p, y) in enumerate(zip(probs, preds)):
        e = entropy_of_probs(p)
        scored.append((y, e, i))
    take = []
    for y in (0,1,2):
        cand = [(e, i) for (yy, e, i) in scored if yy == y]
        cand.sort(key=lambda x: x[0], reverse=True)
        need = quota[y]
        take.extend([i for (_, i) in cand[:need]])
    return take

def top_up_by_entropy(pool, model, tokenizer, need, batch_size, max_len):
    if need <= 0 or len(pool) == 0:
        return [], pool
    idxs_extra = entropy_select(
        unlabeled_pool=pool,
        model=model,
        tokenizer=tokenizer,
        batch_size=batch_size,
        max_len=max_len,
        budget_b=need,
        confidence_filter=None
    )
    extra_rows, pool = split_by_indices(pool, idxs_extra)
    return extra_rows, pool

def main():
    parser = argparse.ArgumentParser(description="Experiment 2 runner (imbalanced) for BERT + Active Learning")
    # Paths
    parser.add_argument("--data_dir", type=str, default="data/ratio_1_1_8",
                        help="Directory containing init_labeled.jsonl and unlabeled_pool.jsonl")
    parser.add_argument("--val_path", type=str, default="data/validation.jsonl")
    parser.add_argument("--test_path", type=str, default="data/test.jsonl")
    parser.add_argument("--model_dir", default="/mnt/parscratch/users/acs24jw/models/roberta-base_model")
    parser.add_argument("--log_dir",   default="/users/acs24jw/nli_project/mis_exp/logs_exp2/RBV")
    parser.add_argument("--out_dir", type=str,
                        default="/users/acs24jw/nli_project/mis_exp/outputs_exp2")

    # Train & strategy
    parser.add_argument("--strategy", type=str, default="entropy",
                        choices=["random", "entropy", "hyp_concat", "short_simple"])
    parser.add_argument("--mode", type=str, default="S1", choices=["S1", "S2"],
                        help="S1=natural; S2=class quota control")
    parser.add_argument("--ratio", type=str, default="1:1:8", help="Target class ratio E:N:C")
    parser.add_argument("--b", type=int, default=100)
    parser.add_argument("--R", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_len", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fp16", action="store_true")

    # Strategy options
    parser.add_argument("--confidence_filter", type=float, default=None)

    # LLM options
    parser.add_argument("--llm_model_dir", type=str,
                        default="/mnt/parscratch/users/acs24jw/llmmodels/Qwen2.5-7B-Instruct/Qwen2.5-7B-Instruct")
    parser.add_argument("--llm_max_trials", type=int, default=400,
                        help="More trials for S2 quotas")

    args = parser.parse_args()
    set_seed(args.seed)

    # Expand & ensure dirs
    args.model_dir = os.path.expanduser(args.model_dir)
    ensure_dir(args.log_dir)
    ensure_dir(args.out_dir)

    # Load data
    init_path = os.path.join(args.data_dir, "init_labeled.jsonl")
    pool_path = os.path.join(args.data_dir, "unlabeled_pool.jsonl")
    train = load_jsonl(init_path)
    pool = load_jsonl(pool_path)
    val = load_jsonl(args.val_path)
    test = load_jsonl(args.test_path)

    # Model + tokenizer
    clf = Classifier(model_dir=args.model_dir, num_labels=3, fp16=args.fp16)
    tokenizer = clf.get_tokenizer()

    # LLM (when needed)
    llm_client = None
    if args.strategy in ("hyp_concat", "short_simple"):
        llm_client = QwenJudge(model_dir=args.llm_model_dir, fp16=True)

    # Tagging
    run_tag = f"roberta_{args.strategy}_{args.mode}_{args.ratio.replace(':','-')}"
    strat_out_dir = os.path.join(args.out_dir, run_tag)
    ensure_dir(strat_out_dir)

    log_csv = os.path.join(args.log_dir, f"{run_tag}.csv")
    if not os.path.exists(log_csv):
        with open(log_csv, "w", encoding="utf-8", newline="") as f:
            csv.writer(f).writerow(["round","labeled_size","added_count","accuracy","macro_f1","precision","recall"])

    # Dataloaders
    train_ds = NliDataset(train, tokenizer, max_len=args.max_len)
    val_ds = NliDataset(val, tokenizer, max_len=args.max_len)
    test_ds = NliDataset(test, tokenizer, max_len=args.max_len)

    train_loader = build_loader(train_ds, batch_size=args.batch_size, shuffle=True,
                                tokenizer=tokenizer, max_len=args.max_len)
    val_loader = build_loader(val_ds, batch_size=args.batch_size, shuffle=False,
                              tokenizer=tokenizer, max_len=args.max_len)
    test_loader = build_loader(test_ds, batch_size=args.batch_size, shuffle=False,
                               tokenizer=tokenizer, max_len=args.max_len)

    # Round 0
    print("== Round 0: train on initial labeled set (RoBERTa) ==")
    _ = train_with_early_stop(clf, train_loader, val_loader,
                              epochs=args.epochs, patience=args.patience, lr=args.lr)
    metrics = clf.evaluate(val_loader)
    with open(log_csv, "a", encoding="utf-8", newline="") as f:
        csv.writer(f).writerow([0, len(train), 0, metrics["accuracy"], metrics["macro_f1"],
                                metrics["precision"], metrics["recall"]])

    target_ratio = parse_ratio(args.ratio)

    for r in range(1, args.R + 1):
        print(f"\n== Round {r}/{args.R} strategy: {args.strategy} mode: {args.mode} ratio: {args.ratio} ==")
        picked_rows = []

        if args.mode == "S1":
            # 与实验1一致：先按策略选 b 条；若是 LLM 策略不足则 entropy 补齐
            idxs, new_samples = select(
                strategy=args.strategy,
                unlabeled_pool=pool,
                model=clf,
                tokenizer=tokenizer,
                batch_size=args.batch_size,
                max_len=args.max_len,
                budget_b=args.b,
                seed=args.seed + r,
                llm_client=llm_client,
                confidence_filter=args.confidence_filter,
                llm_max_trials=args.llm_max_trials,
            )
            if new_samples: picked_rows.extend(new_samples)
            need = args.b - len(picked_rows)
            if need > 0:
                extra_rows, pool = top_up_by_entropy(pool, clf, tokenizer, need, args.batch_size, args.max_len)
                picked_rows.extend(extra_rows)
            if not picked_rows and idxs:
                rows, pool = split_by_indices(pool, idxs)
                picked_rows.extend(rows)

        else:  # S2 控类配额
            quota = compute_quota(args.b, target_ratio)  # [E,N,C]
            if args.strategy in ("random", "entropy"):
                probs, preds = predict_probs_and_labels(clf, tokenizer, pool,
                                                        batch_size=args.batch_size, max_len=args.max_len)
                if args.strategy == "random":
                    take_idxs = s2_select_pool_random(pool, preds, quota, seed=args.seed + r)
                else:
                    take_idxs = s2_select_pool_entropy(pool, probs, preds, quota)
                rows, pool = split_by_indices(pool, take_idxs)
                picked_rows.extend(rows)
            else:
                # 先“过量”生成（最多 6×b），然后按标签做配额，不足再从 pool-entropy 分桶补齐
                over_budget = min(args.b * 6, args.llm_max_trials)
                idxs, new_samples = select(
                    strategy=args.strategy,
                    unlabeled_pool=pool,
                    model=clf,
                    tokenizer=tokenizer,
                    batch_size=args.batch_size,
                    max_len=args.max_len,
                    budget_b=over_budget,
                    seed=args.seed + r,
                    llm_client=llm_client,
                    confidence_filter=args.confidence_filter,
                    llm_max_trials=args.llm_max_trials,
                )
                # 期望 new_samples 含字段 'label' ∈ {0,1,2}
                by_label = {0: [], 1: [], 2: []}
                for ex in (new_samples or []):
                    lb = ex.get("label", None)
                    if lb in (0,1,2):
                        by_label[lb].append(ex)
                # 先按配额从生成样本中拿
                for y in (0,1,2):
                    need_y = quota[y]
                    take = by_label[y][:need_y]
                    picked_rows.extend(take)
                    quota[y] -= len(take)
                # 生成不足，用 pool-entropy 按类补齐
                remaining_need = sum(max(0, q) for q in quota)
                if remaining_need > 0 and len(pool) > 0:
                    probs, preds = predict_probs_and_labels(clf, tokenizer, pool,
                                                            batch_size=args.batch_size, max_len=args.max_len)
                    # 依次对每一类补齐
                    extra_take = []
                    for y in (0,1,2):
                        need_y = max(0, quota[y])
                        if need_y == 0: continue
                        # 取该类中按熵降序的前 need_y
                        cand = [(entropy_of_probs(probs[i]), i) for i in range(len(pool)) if preds[i] == y]
                        cand.sort(key=lambda x: x[0], reverse=True)
                        extra_take.extend([i for (_, i) in cand[:need_y]])
                    rows, pool = split_by_indices(pool, extra_take)
                    picked_rows.extend(rows)

        if not picked_rows:
            raise RuntimeError("No samples were added this round.")

        # merge & save
        before = len(train)
        train.extend(picked_rows)
        added = len(train) - before
        save_jsonl(os.path.join(strat_out_dir, f"round_{r}_train.jsonl"), train)
        save_jsonl(os.path.join(strat_out_dir, f"round_{r}_pool_remaining.jsonl"), pool)

        # retrain + dev eval
        train_ds = NliDataset(train, tokenizer, max_len=args.max_len)
        train_loader = build_loader(train_ds, batch_size=args.batch_size, shuffle=True,
                                    tokenizer=tokenizer, max_len=args.max_len)
        _ = train_with_early_stop(clf, train_loader, val_loader,
                                  epochs=args.epochs, patience=args.patience, lr=args.lr)
        metrics = clf.evaluate(val_loader)
        with open(log_csv, "a", encoding="utf-8", newline="") as f:
            csv.writer(f).writerow([r, len(train), added, metrics["accuracy"], metrics["macro_f1"],
                                    metrics["precision"], metrics["recall"]])

    # final test
    final_test = clf.evaluate(test_loader)
    with open(os.path.join(strat_out_dir, "final_test.json"), "w", encoding="utf-8") as f:
        json.dump(final_test, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()
