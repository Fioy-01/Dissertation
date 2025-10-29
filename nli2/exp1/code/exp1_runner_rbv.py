import os
import csv
import json
import argparse

from transformers.utils import logging as hf_logging
hf_logging.set_verbosity_error()

from datasets import load_jsonl, save_jsonl, split_by_indices, NliDataset
from models import Classifier
from train_utils import set_seed, build_loader, train_with_early_stop
from sampling import select, entropy_select
from llm_qwen import QwenJudge


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def main():
    parser = argparse.ArgumentParser(description="Experiment 1 runner (1:1:1) for RoBERTa-base + Active Learning")

    parser.add_argument("--data_dir", type=str, default="data/ratio_1_1_1")
    parser.add_argument("--val_path", type=str, default="data/validation.jsonl")
    parser.add_argument("--test_path", type=str, default="data/test.jsonl")
    parser.add_argument("--model_dir", type=str,
                        default="/mnt/parscratch/users/acs24jw/models/roberta-base_model")
    parser.add_argument("--log_dir", type=str,
                        default="/users/acs24jw/nli_project/mis_exp/logs/RBV")  # RBV
    parser.add_argument("--out_dir", type=str,
                        default="/users/acs24jw/nli_project/mis_exp/outputs")

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

    parser.add_argument("--confidence_filter", type=float, default=None)

    parser.add_argument("--llm_model_dir", type=str,
                        default="/mnt/parscratch/users/acs24jw/llmmodels/Qwen2.5-7B-Instruct/Qwen2.5-7B-Instruct")
    parser.add_argument("--llm_max_trials", type=int, default=200)

    args = parser.parse_args()
    set_seed(args.seed)

    args.model_dir = os.path.expanduser(args.model_dir)
    ensure_dir(args.log_dir)
    ensure_dir(args.out_dir)

    init_path = os.path.join(args.data_dir, "init_labeled.jsonl")
    pool_path = os.path.join(args.data_dir, "unlabeled_pool.jsonl")
    train = load_jsonl(init_path)
    pool = load_jsonl(pool_path)
    val = load_jsonl(args.val_path)
    test = load_jsonl(args.test_path)

    clf = Classifier(model_dir=args.model_dir, num_labels=3, fp16=args.fp16)
    tokenizer = clf.get_tokenizer()

    llm_client = None
    if args.strategy in ("hyp_concat", "short_simple"):
        llm_client = QwenJudge(model_dir=args.llm_model_dir, fp16=True)

    run_tag = f"roberta_{args.strategy}"
    strat_out_dir = os.path.join(args.out_dir, run_tag)
    ensure_dir(strat_out_dir)

    log_csv = os.path.join(args.log_dir, f"{run_tag}.csv")
    if not os.path.exists(log_csv):
        with open(log_csv, "w", encoding="utf-8", newline="") as f:
            csv.writer(f).writerow(["round", "labeled_size", "added_count",
                                    "accuracy", "macro_f1", "precision", "recall"])

    train_ds = NliDataset(train, tokenizer, max_len=args.max_len)
    val_ds = NliDataset(val, tokenizer, max_len=args.max_len)
    test_ds = NliDataset(test, tokenizer, max_len=args.max_len)

    train_loader = build_loader(train_ds, batch_size=args.batch_size, shuffle=True,
                                tokenizer=tokenizer, max_len=args.max_len)
    val_loader = build_loader(val_ds, batch_size=args.batch_size, shuffle=False,
                              tokenizer=tokenizer, max_len=args.max_len)
    test_loader = build_loader(test_ds, batch_size=args.batch_size, shuffle=False,
                               tokenizer=tokenizer, max_len=args.max_len)

    print("== Round 0: train on initial labeled set (RoBERTa) ==")
    _ = train_with_early_stop(clf, train_loader, val_loader,
                              epochs=args.epochs, patience=args.patience, lr=args.lr)
    metrics = clf.evaluate(val_loader)
    with open(log_csv, "a", encoding="utf-8", newline="") as f:
        csv.writer(f).writerow([0, len(train), 0, metrics["accuracy"],
                                metrics["macro_f1"], metrics["precision"], metrics["recall"]])

    for r in range(1, args.R + 1):
        print(f"\n== Round {r}/{args.R} strategy: {args.strategy} ==")
        idxs, new_samples = select(strategy=args.strategy, unlabeled_pool=pool, model=clf,
                                   tokenizer=tokenizer, batch_size=args.batch_size,
                                   max_len=args.max_len, budget_b=args.b, seed=args.seed + r,
                                   llm_client=llm_client, confidence_filter=args.confidence_filter,
                                   llm_max_trials=args.llm_max_trials)

        picked_rows = []
        if new_samples:
            picked_rows.extend(new_samples)

        need = args.b - len(picked_rows)
        if need > 0:
            idxs_extra = entropy_select(unlabeled_pool=pool, model=clf, tokenizer=tokenizer,
                                        batch_size=args.batch_size, max_len=args.max_len,
                                        budget_b=need, confidence_filter=None)
            extra_rows, pool = split_by_indices(pool, idxs_extra)
            picked_rows.extend(extra_rows)

        if not picked_rows and idxs:
            rows, pool = split_by_indices(pool, idxs)
            picked_rows.extend(rows)

        before = len(train)
        train.extend(picked_rows)
        added = len(train) - before
        save_jsonl(os.path.join(strat_out_dir, f"round_{r}_train.jsonl"), train)
        save_jsonl(os.path.join(strat_out_dir, f"round_{r}_pool_remaining.jsonl"), pool)

        train_ds = NliDataset(train, tokenizer, max_len=args.max_len)
        train_loader = build_loader(train_ds, batch_size=args.batch_size, shuffle=True,
                                    tokenizer=tokenizer, max_len=args.max_len)
        _ = train_with_early_stop(clf, train_loader, val_loader,
                                  epochs=args.epochs, patience=args.patience, lr=args.lr)
        metrics = clf.evaluate(val_loader)
        with open(log_csv, "a", encoding="utf-8", newline="") as f:
            csv.writer(f).writerow([r, len(train), added, metrics["accuracy"],
                                    metrics["macro_f1"], metrics["precision"], metrics["recall"]])

    final_test = clf.evaluate(test_loader)
    with open(os.path.join(strat_out_dir, "final_test.json"), "w", encoding="utf-8") as f:
        json.dump(final_test, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()

