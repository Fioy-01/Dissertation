import os
import csv
import json
import argparse

from transformers.utils import logging as hf_logging
hf_logging.set_verbosity_error()  # quiet HF logs

from datasets import load_jsonl, save_jsonl, split_by_indices, NliDataset
from models import Classifier
from train_utils import set_seed, build_loader, train_with_early_stop
from sampling import select, entropy_select
from llm_qwen import QwenJudge


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def main():
    parser = argparse.ArgumentParser(description="Experiment 1 runner (1:1:1) for DistilBERT + Active Learning")

    # Paths
    parser.add_argument("--data_dir", type=str, default="data/ratio_1_1_1",
                        help="Directory containing init_labeled.jsonl and unlabeled_pool.jsonl")
    parser.add_argument("--val_path", type=str, default="data/validation.jsonl")
    parser.add_argument("--test_path", type=str, default="data/test.jsonl")
    parser.add_argument("--model_dir", type=str,
                        default="/mnt/parscratch/users/acs24jw/models/distilbert-base-uncased_model",
                        help="Local DistilBERT classifier directory")
    parser.add_argument("--log_dir", type=str,
                        default="/users/acs24jw/nli_project/mis_exp/logs/DBT",
                        help="Directory to save per-round metrics CSV (DistilBERT = DBT)")
    parser.add_argument("--out_dir", type=str,
                        default="/users/acs24jw/nli_project/mis_exp/outputs",
                        help="Directory to save augmented training sets and final results")

    # Training & strategy
    parser.add_argument("--strategy", type=str, default="entropy",
                        choices=["random", "entropy", "hyp_concat", "short_simple"],
                        help="Active learning strategy")
    parser.add_argument("--b", type=int, default=100, help="Acquisition/generation budget per round")
    parser.add_argument("--R", type=int, default=4, help="Number of active learning rounds")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_len", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fp16", action="store_true")

    # Strategy options
    parser.add_argument("--confidence_filter", type=float, default=None,
                        help="Optional max-probability filter for entropy selection (e.g., 0.85). None disables filtering.")

    # LLM (only for hyp_concat / short_simple)
    parser.add_argument("--llm_model_dir", type=str,
                        default="/mnt/parscratch/users/acs24jw/llmmodels/Qwen2.5-7B-Instruct/Qwen2.5-7B-Instruct",
                        help="Local Qwen2.5-7B-Instruct path for generation/judging")
    parser.add_argument("--llm_max_trials", type=int, default=200,
                        help="Max attempts when generating short & simple samples")

    args = parser.parse_args()
    set_seed(args.seed)

    # Expand ~ if any and ensure dirs
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

    # Model & tokenizer (DistilBERT)
    clf = Classifier(model_dir=args.model_dir, num_labels=3, fp16=args.fp16)
    tokenizer = clf.get_tokenizer()

    # LLM client for LLM strategies
    llm_client = None
    if args.strategy in ("hyp_concat", "short_simple"):
        llm_client = QwenJudge(model_dir=args.llm_model_dir, fp16=True)

    # Per-strategy output subdir; tag fixed to "distil"
    run_tag = f"distil_{args.strategy}"
    strat_out_dir = os.path.join(args.out_dir, run_tag)
    ensure_dir(strat_out_dir)

    # Metrics CSV with added_count
    log_csv = os.path.join(args.log_dir, f"{run_tag}.csv")
    if not os.path.exists(log_csv):
        with open(log_csv, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["round", "labeled_size", "added_count", "accuracy", "macro_f1", "precision", "recall"])

    # Build loaders (batch tokenization in collate_fn)
    train_ds = NliDataset(train, tokenizer, max_len=args.max_len)
    val_ds = NliDataset(val, tokenizer, max_len=args.max_len)
    test_ds = NliDataset(test, tokenizer, max_len=args.max_len)

    train_loader = build_loader(train_ds, batch_size=args.batch_size, shuffle=True,
                                tokenizer=tokenizer, max_len=args.max_len)
    val_loader = build_loader(val_ds, batch_size=args.batch_size, shuffle=False,
                              tokenizer=tokenizer, max_len=args.max_len)
    test_loader = build_loader(test_ds, batch_size=args.batch_size, shuffle=False,
                               tokenizer=tokenizer, max_len=args.max_len)

    # Round 0: initial training + dev eval
    print("== Round 0: train on initial labeled set (DistilBERT) ==")
    _ = train_with_early_stop(
        clf, train_loader, val_loader,
        epochs=args.epochs, patience=args.patience, lr=args.lr
    )
    metrics = clf.evaluate(val_loader)
    print(f"[Round 0] val: {metrics}")
    with open(log_csv, "a", encoding="utf-8", newline="") as f:
        csv.writer(f).writerow([0, len(train), 0, metrics["accuracy"], metrics["macro_f1"],
                                metrics["precision"], metrics["recall"]])

    # Active learning rounds
    for r in range(1, args.R + 1):
        print(f"\n== Round {r}/{args.R} strategy: {args.strategy} (DistilBERT) ==")

        # Strategy may return indices (pool) or new_samples (LLM)
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

        # 1) take LLM-generated/judged samples first
        if new_samples is not None and len(new_samples) > 0:
            picked_rows.extend(new_samples)

        # 2) top up from pool via entropy to fill budget
        need = args.b - len(picked_rows)
        if need > 0:
            idxs_extra = entropy_select(
                unlabeled_pool=pool,
                model=clf,
                tokenizer=tokenizer,
                batch_size=args.batch_size,
                max_len=args.max_len,
                budget_b=need,
                confidence_filter=None
            )
            extra_rows, pool = split_by_indices(pool, idxs_extra)
            picked_rows.extend(extra_rows)

        # 3) if selector returned pure pool indices and nothing picked yet
        if len(picked_rows) == 0 and idxs is not None and len(idxs) > 0:
            rows, pool = split_by_indices(pool, idxs)
            picked_rows.extend(rows)

        if len(picked_rows) == 0:
            raise RuntimeError("No samples were added this round (new_samples=0 and no pool fallback).")

        # Merge into training set and save snapshot
        before = len(train)
        train.extend(picked_rows)
        after = len(train)
        added = after - before
        print(f"Added {added} samples. New train size = {after}")
        save_jsonl(os.path.join(strat_out_dir, f"round_{r}_train.jsonl"), train)

        # Save remaining pool
        save_jsonl(os.path.join(strat_out_dir, f"round_{r}_pool_remaining.jsonl"), pool)

        # Retrain and eval on dev
        train_ds = NliDataset(train, tokenizer, max_len=args.max_len)
        train_loader = build_loader(train_ds, batch_size=args.batch_size, shuffle=True,
                                    tokenizer=tokenizer, max_len=args.max_len)
        _ = train_with_early_stop(
            clf, train_loader, val_loader,
            epochs=args.epochs, patience=args.patience, lr=args.lr
        )
        metrics = clf.evaluate(val_loader)
        print(f"[Round {r}] val: {metrics}")
        with open(log_csv, "a", encoding="utf-8", newline="") as f:
            csv.writer(f).writerow([r, len(train), added, metrics["accuracy"], metrics["macro_f1"],
                                    metrics["precision"], metrics["recall"]])

    # Final test
    final_test = clf.evaluate(test_loader)
    print(f"\n== Final Test (DistilBERT) ==\n{final_test}")
    with open(os.path.join(strat_out_dir, "final_test.json"), "w", encoding="utf-8") as f:
        json.dump(final_test, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()

