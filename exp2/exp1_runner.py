import os
import csv
import json
import argparse
from typing import List, Dict

from datasets import load_jsonl, save_jsonl, split_by_indices, NliDataset, LABEL2ID, ID2LABEL
from models import Classifier
from train_utils import set_seed, build_loader, train_with_early_stop
from sampling import select
from llm_qwen import QwenJudge


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)


def main():
    parser = argparse.ArgumentParser()
    # 路径
    parser.add_argument("--data_dir", type=str, default="data/ratio_1_1_1")
    parser.add_argument("--val_path", type=str, default="data/validation.jsonl")
    parser.add_argument("--test_path", type=str, default="data/test.jsonl")
    parser.add_argument("--model_dir", type=str, default="~/models/bert-base-uncased_model")
    parser.add_argument("--log_dir", type=str, default="logs")
    parser.add_argument("--out_dir", type=str, default="outputs")
    # 训练与策略
    parser.add_argument("--strategy", type=str, default="entropy",
                        choices=["random", "entropy", "hyp_concat", "short_simple"])
    parser.add_argument("--b", type=int, default=100)
    parser.add_argument("--R", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_len", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fp16", action="store_true")
    # 采样细节
    parser.add_argument("--confidence_filter", type=float, default=None)  # e.g., 0.85
    # LLM
    parser.add_argument("--llm_model_dir", type=str, default="/mnt/parscratch/users/acs24jw/llmmodels/Qwen2.5-7B-Instruct/Qwen2.5-7B-Instruct")
    parser.add_argument("--llm_max_trials", type=int, default=200)

    args = parser.parse_args()
    set_seed(args.seed)

    # 路径展开
    args.model_dir = os.path.expanduser(args.model_dir)

    # 读数据
    init_path = os.path.join(args.data_dir, "init_labeled.jsonl")
    pool_path = os.path.join(args.data_dir, "unlabeled_pool.jsonl")
    train = load_jsonl(init_path)
    pool = load_jsonl(pool_path)
    val = load_jsonl(args.val_path)
    test = load_jsonl(args.test_path)

    # 模型
    clf = Classifier(model_dir=args.model_dir, num_labels=3, fp16=args.fp16)
    tokenizer = clf.get_tokenizer()

    # LLM（仅当策略需要）
    llm_client = None
    if args.strategy in ("hyp_concat", "short_simple"):
        llm_client = QwenJudge(model_dir=args.llm_model_dir, fp16=True)

    # 输出目录
    run_tag = f"bert_{args.strategy}"
    strat_out_dir = os.path.join(args.out_dir, run_tag)
    ensure_dir(strat_out_dir)
    ensure_dir(args.log_dir)

    # 日志 CSV
    log_csv = os.path.join(args.log_dir, f"{run_tag}.csv")
    if not os.path.exists(log_csv):
        with open(log_csv, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["round", "labeled_size", "accuracy", "macro_f1", "precision", "recall"])

    # Round 0：初始训练与评估
    print("== Round 0: train on initial labeled set ==")
    train_ds = NliDataset(train, tokenizer, max_len=args.max_len)
    val_ds = NliDataset(val, tokenizer, max_len=args.max_len)
    test_ds = NliDataset(test, tokenizer, max_len=args.max_len)

    train_loader = build_loader(train_ds, batch_size=args.batch_size, shuffle=True, tokenizer=tokenizer)
    val_loader = build_loader(val_ds, batch_size=args.batch_size, shuffle=False, tokenizer=tokenizer)
    test_loader = build_loader(test_ds, batch_size=args.batch_size, shuffle=False, tokenizer=tokenizer)

    _ = train_with_early_stop(
        clf, train_loader, val_loader,
        epochs=args.epochs, patience=args.patience, lr=args.lr
    )
    metrics = clf.evaluate(val_loader)
    print(f"[Round 0] val: {metrics}")
    with open(log_csv, "a", encoding="utf-8", newline="") as f:
        csv.writer(f).writerow([0, len(train), metrics["accuracy"], metrics["macro_f1"], metrics["precision"], metrics["recall"]])

    # 主循环：R 轮主动学习
    for r in range(1, args.R + 1):
        print(f"\n== Round {r}/{args.R} sampling strategy: {args.strategy} ==")

        # 选样（可能返回索引，也可能直接返回生成样本）
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

        picked_rows = []
        if idxs is not None:
            # 来自 pool 的选样：揭示标签并从 pool 移除
            picked_rows, pool = split_by_indices(pool, idxs)
            # pool 落盘（可选）
            save_jsonl(os.path.join(strat_out_dir, f"round_{r}_pool_remaining.jsonl"), pool)
        elif new_samples is not None:
            picked_rows = new_samples
        else:
            raise RuntimeError("Neither indices nor new_samples returned.")

        # 将选样并入训练集
        before = len(train)
        train.extend(picked_rows)
        after = len(train)
        print(f"Added {after - before} samples into training set. New train size = {after}")

        # 当前轮的“增强后训练集”落盘（便于追溯）
        save_jsonl(os.path.join(strat_out_dir, f"round_{r}_train.jsonl"), train)

        # 重新训练并在 dev 上评估
        train_ds = NliDataset(train, tokenizer, max_len=args.max_len)
        train_loader = build_loader(train_ds, batch_size=args.batch_size, shuffle=True, tokenizer=tokenizer)
        _ = train_with_early_stop(
            clf, train_loader, val_loader,
            epochs=args.epochs, patience=args.patience, lr=args.lr
        )
        metrics = clf.evaluate(val_loader)
        print(f"[Round {r}] val: {metrics}")
        with open(log_csv, "a", encoding="utf-8", newline="") as f:
            csv.writer(f).writerow([r, len(train), metrics["accuracy"], metrics["macro_f1"], metrics["precision"], metrics["recall"]])

    # 最终：在 test 上评估
    final_test = clf.evaluate(test_loader)
    print(f"\n== Final Test ==\n{final_test}")
    with open(os.path.join(strat_out_dir, "final_test.json"), "w", encoding="utf-8") as f:
        json.dump(final_test, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
