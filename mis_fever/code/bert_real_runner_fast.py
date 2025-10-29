# -*- coding: utf-8 -*-
"""
bert_real_runner_fast.py
仅记录 round_log.csv；支持 5 策略（alvin / short_simple / long_simple / difficulty / misclassified）
与 3 模式（uncontrolled / controlled-balanced / controlled-optimal）。
为提速：单模型单种子、极小训练轮次与长度、默认跳过几何覆盖计算（rho=0）。
已补齐 QwenJudge 初始化与 --qwen_dir / --enable_llm，用于真实验证 short/long 生成策略。
"""

import os
import argparse
import time
from copy import deepcopy
from typing import List, Dict, Tuple

import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix
import torch

from datasets import load_jsonl, NliDataset, LABEL2ID, ID2LABEL
from models import Classifier
from train_utils import set_seed, build_loader, train_with_early_stop
from sampling import select as al_select
from llm_qwen import QwenJudge  # ★ 新增：用于两种生成策略

# ================= 工具函数 =================

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def timestamp():
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

ROUND_LOG_HEADER = [
    "timestamp","strategy","mode","round","labeled_total","B",
    "minority_class","minority_f1","minority_recall","minority_precision",
    "macro_f1","accuracy","macro_precision","macro_recall",
    "pi_e","pi_n","pi_c",
    "n_e","n_n","n_c",
    "kappa_uniform","kappa_invfreq","rho_t_max","rho_t_p95"
]

def log_round_metrics(round_log_path, data_dict):
    new_file = not os.path.exists(round_log_path)
    with open(round_log_path, "a", encoding="utf-8") as fw:
        if new_file:
            fw.write(",".join(ROUND_LOG_HEADER) + "\n")
        fw.write(",".join(map(str, [
            timestamp(),
            data_dict["strat"],
            data_dict["mode"],
            data_dict["round"],
            data_dict["labeled_total"],
            data_dict["B"],
            data_dict["minority_class"],
            f"{data_dict['minority_f1']:.4f}",
            f"{data_dict['minority_recall']:.4f}",
            f"{data_dict['minority_precision']:.4f}",
            f"{data_dict['macro_f1']:.4f}",
            f"{data_dict['accuracy']:.4f}",
            f"{data_dict['macro_precision']:.4f}",
            f"{data_dict['macro_recall']:.4f}",
            f"{data_dict['pi_e']:.4f}",
            f"{data_dict['pi_n']:.4f}",
            f"{data_dict['pi_c']:.4f}",
            data_dict["n_e"],
            data_dict["n_n"],
            data_dict["n_c"],
            f"{data_dict['kappa_uniform']:.6f}",
            f"{data_dict['kappa_invfreq']:.6f}",
            f"{data_dict['rho_t_max']:.6f}",
            f"{data_dict['rho_t_p95']:.6f}",
        ])) + "\n")

def strat_name_clean(s: str) -> str:
    return s.replace("/", "-").replace("&", "_").replace(" ", "").lower()

def load_split(init_dir: str) -> Tuple[List[Dict], List[Dict]]:
    L = load_jsonl(os.path.join(init_dir, "init_labeled.jsonl"))
    U = load_jsonl(os.path.join(init_dir, "unlabeled_pool.jsonl"))
    return L, U

def load_eval_sets(val_path: str, test_path: str) -> Tuple[List[Dict], List[Dict]]:
    V = load_jsonl(val_path)
    T = load_jsonl(test_path)
    return V, T

def build_dataset_and_loader(rows: List[Dict], tokenizer, batch_size: int, max_len: int, shuffle: bool):
    ds = NliDataset(rows, tokenizer=tokenizer, max_len=max_len)
    return ds, build_loader(ds, batch_size=batch_size, shuffle=shuffle, tokenizer=tokenizer, max_len=max_len)

def compute_per_class_metrics(golds: List[int], preds: List[int], n_classes: int = 3):
    acc = accuracy_score(golds, preds)
    prec, rec, f1, support = precision_recall_fscore_support(
        golds, preds, labels=list(range(n_classes)), average=None, zero_division=0
    )
    macro_prec = float(np.mean(prec))
    macro_rec = float(np.mean(rec))
    macro_f1 = float(np.mean(f1))
    cm = confusion_matrix(golds, preds, labels=list(range(n_classes)))
    return {
        "accuracy": float(acc),
        "macro_precision": macro_prec,
        "macro_recall": macro_rec,
        "macro_f1": float(macro_f1),
        "per_class_precision": prec.tolist(),
        "per_class_recall": rec.tolist(),
        "per_class_f1": f1.tolist(),
        "support": support.tolist(),
        "confusion_matrix": cm.tolist(),
    }

def eval_model_on_split(clf: Classifier, rows: List[Dict], batch_size: int, max_len: int):
    tok = clf.get_tokenizer()
    _, loader = build_dataset_and_loader(rows, tok, batch_size=batch_size, max_len=max_len, shuffle=False)
    clf.model.eval()
    preds, golds = [], []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(clf.device)
            attention_mask = batch["attention_mask"].to(clf.device)
            labels = batch["labels"].to(clf.device)
            out = clf.model(input_ids=input_ids, attention_mask=attention_mask)
            p = torch.argmax(out.logits, dim=-1)
            preds.extend(p.detach().cpu().numpy().tolist())
            golds.extend(labels.detach().cpu().numpy().tolist())
    return compute_per_class_metrics(golds, preds, n_classes=3)

def majority_minority_classes(counts: Dict[str, int]):
    if not counts:
        return [], []
    min_cnt = min(counts.values())
    max_cnt = max(counts.values())
    mins = [k for k, v in counts.items() if v == min_cnt]
    maxs = [k for k, v in counts.items() if v == max_cnt]
    return mins, maxs

def count_by_label(rows: List[Dict]) -> Dict[str, int]:
    cnt = {"entailment": 0, "neutral": 0, "contradiction": 0}
    for r in rows:
        lab = r.get("label")
        if isinstance(lab, int):
            lab = ID2LABEL[lab]
        if lab in cnt:
            cnt[lab] += 1
    return cnt

# 轻量平衡配额（基于预测标签）
def apply_balanced_quota_with_predictions(sorted_indices: List[int],
                                          probs_on_pool: np.ndarray,
                                          B: int,
                                          start_major: str = "entailment") -> List[int]:
    base = B // 3
    extra = B - base * 3
    order = ["entailment", "neutral", "contradiction"]
    if start_major in order:
        while order[0] != start_major:
            order.append(order.pop(0))
    target = {lab: base for lab in order}
    for i in range(extra):
        target[order[i]] += 1
    y_pred = probs_on_pool.argmax(axis=1)
    buckets = {"entailment": [], "neutral": [], "contradiction": []}
    for idx in sorted_indices:
        lab = ID2LABEL[int(y_pred[idx])]
        buckets[lab].append(idx)
    selected = []
    for lab in order:
        need = target[lab]
        selected.extend(buckets[lab][:need])
    if len(selected) < B:
        taken = set(selected)
        for idx in sorted_indices:
            if len(selected) >= B:
                break
            if idx not in taken:
                selected.append(idx)
    return selected[:B]

# rho 覆盖（极简：置 0）
def compute_coverage_metrics_fast_stub():
    return {"rho_t_max": 0.0, "rho_t_p95": 0.0}

# ================= 主流程 =================

def run(args):
    # 路径与输出
    INIT_DIR = args.init_dir
    VAL_PATH = args.val_path
    TEST_PATH = args.test_path
    MODEL_DIR = args.model_dir
    ensure_dir(args.output_root)

    # 数据
    init_L, init_U = load_split(INIT_DIR)
    val_rows, test_rows = load_eval_sets(VAL_PATH, TEST_PATH)

    # 轻量 tokenizer 引导
    temp_clf = Classifier(MODEL_DIR, num_labels=3, fp16=True)
    tokenizer = temp_clf.get_tokenizer()
    del temp_clf

    # 策略与模式
    strategies = [strat_name_clean(s) for s in args.strategies if s.strip()]
    modes = [m.lower() for m in args.modes]
    balance_rotation = ["entailment", "neutral", "contradiction"]
    balance_rot_ptr = 0

    # Qwen（真正启用生成所必需）
    QWEN_DIR = args.qwen_dir if args.qwen_dir else None
    llm_client = None
    if args.enable_llm and QWEN_DIR is not None:
        llm_client = QwenJudge(QWEN_DIR, fp16=True)

    for strat in strategies:
        for mode in modes:
            exp_name = f"BERT_fast_{strat}_{mode}"
            out_dir = os.path.join(args.output_root, exp_name)
            ensure_dir(out_dir)
            round_log_path = os.path.join(out_dir, "round_log.csv")

            # 初始化
            L = deepcopy(init_L)
            U = deepcopy(init_U)

            for t in range(1, args.rounds + 1):
                print(f"\n=== [{exp_name}] Round {t}/{args.rounds} ===")

                # (A) 训练（单种子）
                set_seed(args.seed + t)
                clf = Classifier(MODEL_DIR, num_labels=3, fp16=True)
                tok_local = clf.get_tokenizer()
                _, train_loader = build_dataset_and_loader(L, tok_local, batch_size=args.batch_size,
                                                           max_len=args.max_len, shuffle=True)
                _, val_loader = build_dataset_and_loader(val_rows, tok_local, batch_size=args.batch_size,
                                                         max_len=args.max_len, shuffle=False)
                train_with_early_stop(
                    clf, train_loader, val_loader,
                    epochs=args.epochs, patience=args.patience, lr=args.lr,
                    total_steps_hint=len(train_loader) * max(1, args.epochs)
                )

                # (B) 验证集评估
                m_val = eval_model_on_split(clf, val_rows, batch_size=args.batch_size, max_len=args.max_len)
                macro_f1 = m_val["macro_f1"]; acc = m_val["accuracy"]
                macro_prec = m_val["macro_precision"]; macro_rec = m_val["macro_recall"]

                cnt_L = count_by_label(L)
                mins, _ = majority_minority_classes(cnt_L)
                minority = mins[0] if mins else "neutral"
                min_idx = LABEL2ID[minority]
                minority_f1 = float(m_val["per_class_f1"][min_idx])
                minority_rec = float(m_val["per_class_recall"][min_idx])
                minority_prec = float(m_val["per_class_precision"][min_idx])

                # (C) 选样
                B = args.budget
                selected_indices: List[int] = []
                added_rows_generated: List[Dict] = []

                if strat in ("alvin",):
                    idxs, _ = al_select(
                        strategy="alvin",
                        unlabeled_pool=U,
                        model=clf, tokenizer=tok_local,
                        batch_size=args.batch_size, max_len=args.max_len,
                        budget_b=B if mode == "uncontrolled" else len(U),
                        seed=args.seed,
                        labeled_set=L, dynamics=None,
                        low_conf_thresh=0.5, anchors_per_class=10,
                        alpha=2.0, knn_k=8, rng_seed=args.seed,
                        normalize_encodings=False, use_faiss=False, faiss_use_gpu=False,
                    )
                    if mode == "uncontrolled":
                        selected_indices = idxs[:B]
                    else:
                        # 受控：按预测类别配额
                        dsU = NliDataset(U, tokenizer=tok_local, max_len=args.max_len)
                        ldU = build_loader(dsU, batch_size=args.batch_size, shuffle=False,
                                           tokenizer=tok_local, max_len=args.max_len)
                        probs = []
                        clf.model.eval()
                        with torch.no_grad():
                            for batch in ldU:
                                input_ids = batch["input_ids"].to(clf.device)
                                attention_mask = batch["attention_mask"].to(clf.device)
                                out = clf.model(input_ids=input_ids, attention_mask=attention_mask)
                                p = torch.softmax(out.logits, dim=-1).detach().cpu().numpy()
                                probs.append(p)
                        probs = np.concatenate(probs, axis=0) if len(probs) else np.zeros((len(U), 3), dtype=np.float32)

                        if mode == "controlled-balanced":
                            start_major = balance_rotation[balance_rot_ptr % 3]; balance_rot_ptr += 1
                            selected_indices = apply_balanced_quota_with_predictions(
                                sorted_indices=idxs, probs_on_pool=probs, B=B, start_major=start_major
                            )
                        elif mode == "controlled-optimal":
                            # 极简启发式：上一轮 recall 低的类多给配额
                            base = B // 3
                            bonus_all = B - base * 3
                            last_rec = {
                                "entailment": float(m_val["per_class_recall"][LABEL2ID["entailment"]]),
                                "neutral": float(m_val["per_class_recall"][LABEL2ID["neutral"]]),
                                "contradiction": float(m_val["per_class_recall"][LABEL2ID["contradiction"]]),
                            }
                            need = {k: base for k in ["entailment","neutral","contradiction"]}
                            order = sorted(last_rec.items(), key=lambda kv: (1.0 - kv[1]), reverse=True)
                            for i in range(bonus_all):
                                need[order[i % 3][0]] += 1
                            y_pred = probs.argmax(axis=1)
                            buckets = {"entailment": [], "neutral": [], "contradiction": []}
                            for idx in idxs:
                                lab = ID2LABEL[int(y_pred[idx])]
                                buckets[lab].append(idx)
                            chosen = []
                            for lab in ["entailment","neutral","contradiction"]:
                                chosen.extend(buckets[lab][:need[lab]])
                            if len(chosen) < B:
                                taken = set(chosen)
                                for idx in idxs:
                                    if len(chosen) >= B: break
                                    if idx not in taken: chosen.append(idx)
                            selected_indices = chosen[:B]
                        else:
                            selected_indices = idxs[:B]

                elif strat in ("short_simple", "long_simple"):
                    if llm_client is not None:
                        # 真实生成：返回 new_samples（本版不落盘，仅并入 L）
                        idxs_gen, new_samples = al_select(
                            strategy=strat,
                            unlabeled_pool=[],
                            model=None, tokenizer=None,
                            batch_size=args.batch_size, max_len=args.max_len,
                            budget_b=B, seed=args.seed,
                            llm_client=llm_client,  # ★ 传入实例
                            n_votes=8, balance_generated=True, llm_max_trials=1000
                        )
                        added_rows_generated = new_samples or []
                        for r in added_rows_generated:
                            if isinstance(r.get("label", None), str):
                                r["label"] = LABEL2ID[r["label"]]
                        # 生成即视为本轮新增的一部分；若不足 B，可再从池补足
                        rest = max(0, B - len(added_rows_generated))
                        if rest > 0 and len(U) > 0:
                            idxs_pool, _ = al_select(
                                strategy="alvin",
                                unlabeled_pool=U,
                                model=clf, tokenizer=tok_local,
                                batch_size=args.batch_size, max_len=args.max_len,
                                budget_b=rest, seed=args.seed,
                                labeled_set=L, anchors_per_class=10, knn_k=8
                            )
                            selected_indices = idxs_pool[:rest]
                    else:
                        # 未启用 LLM：回退池采样，保证流程可运行
                        idxs, _ = al_select(
                            strategy="alvin",
                            unlabeled_pool=U,
                            model=clf, tokenizer=tok_local,
                            batch_size=args.batch_size, max_len=args.max_len,
                            budget_b=B if mode == "uncontrolled" else len(U),
                            seed=args.seed,
                            labeled_set=L, anchors_per_class=10, knn_k=8
                        )
                        selected_indices = idxs[:B]

                elif strat in ("difficulty", "misclassified"):
                    # 诊断策略：为保证各回合长度一致，快速随机补选 B 个池样本（不做诊断落盘）
                    if len(U) > 0 and B > 0:
                        idxs = list(range(len(U)))
                        rng = np.random.default_rng(args.seed + 1000 + t)
                        rng.shuffle(idxs)
                        selected_indices = idxs[:min(B, len(U))]
                    else:
                        selected_indices = []

                else:
                    raise ValueError(f"Unknown strategy for fast runner: {strat}")

                # (D) Update L/U
                if added_rows_generated:
                    L.extend(added_rows_generated)  # 生成样本直接并入

                added_rows_pool = []
                if selected_indices:
                    sel_set = set(selected_indices)
                    remain = []
                    for i, r in enumerate(U):
                        (added_rows_pool if i in sel_set else remain).append(r)
                    U = remain
                    for r in added_rows_pool:
                        if isinstance(r.get("label", None), str):
                            r["label"] = LABEL2ID[r["label"]]
                    L.extend(added_rows_pool)

                # (E) 覆盖与 kappa（rho 置 0）
                n_counts = count_by_label(L)
                N_total_per_class = count_by_label(init_L + init_U)

                def kappa(weights_mode: str):
                    s = 0.0; wsum = 0.0
                    for k in ["entailment", "neutral", "contradiction"]:
                        Nk = max(1, N_total_per_class.get(k, 1))
                        if weights_mode == "uniform":
                            wk = 1.0 / 3.0
                        else:
                            wk = 1.0 / float(Nk)
                        s += wk * (n_counts.get(k, 0) / float(Nk))
                        wsum += wk
                    return float(s / max(1e-9, wsum))

                kappa_uniform = kappa("uniform")
                kappa_invfreq = kappa("inv")
                cov = {"rho_t_max": 0.0, "rho_t_p95": 0.0} if args.fast_coverage else compute_coverage_metrics_fast_stub()

                # 选择分布（只统计“本轮新增的池样本”的真标签占比；生成样本不计入 pi）
                added_rows = added_rows_pool
                pi_e = pi_n = pi_c = 0.0
                if added_rows:
                    sel_cnt = {"entailment": 0, "neutral": 0, "contradiction": 0}
                    denom = max(1, len(added_rows))
                    for r in added_rows:
                        lab = r.get("label")
                        if isinstance(lab, int):
                            lab = ID2LABEL[lab]
                        if lab in sel_cnt:
                            sel_cnt[lab] += 1
                    pi_e = sel_cnt["entailment"] / denom
                    pi_n = sel_cnt["neutral"] / denom
                    pi_c = sel_cnt["contradiction"] / denom

                # (F) 仅写 round_log.csv
                log_round_metrics(round_log_path, {
                    "strat": strat,
                    "mode": mode,
                    "round": t,
                    "labeled_total": len(L),
                    "B": B,
                    "minority_class": minority,
                    "minority_f1": minority_f1,
                    "minority_recall": minority_rec,
                    "minority_precision": minority_prec,
                    "macro_f1": macro_f1,
                    "accuracy": acc,
                    "macro_precision": macro_prec,
                    "macro_recall": macro_rec,
                    "pi_e": pi_e,
                    "pi_n": pi_n,
                    "pi_c": pi_c,
                    "n_e": n_counts.get("entailment", 0),
                    "n_n": n_counts.get("neutral", 0),
                    "n_c": n_counts.get("contradiction", 0),
                    "kappa_uniform": kappa_uniform,
                    "kappa_invfreq": kappa_invfreq,
                    "rho_t_max": cov["rho_t_max"],
                    "rho_t_p95": cov["rho_t_p95"],
                })

            print(f"[Done] {exp_name} → {round_log_path}")

# ================= 参数 =================

def build_argparser():
    p = argparse.ArgumentParser(description="Fast AL runner (5 strategies × 3 modes) with Qwen generation; only writes round_log.csv")
    # 路径
    p.add_argument("--init_dir", type=str,
                   default="/users/acs24jw/nli_project/mis_fever/data/fever_2000/ratio_real")
    p.add_argument("--val_path", type=str,
                   default="/users/acs24jw/nli_project/mis_fever/data/dev_f.jsonl")
    p.add_argument("--test_path", type=str,
                   default="/users/acs24jw/nli_project/mis_fever/data/test_f.jsonl")
    p.add_argument("--model_dir", type=str,
                   default="/mnt/parscratch/users/acs24jw/models/bert-base-uncased_model")
    p.add_argument("--output_root", type=str,
                   default="/users/acs24jw/nli_project/mis_fever/outputs_fast")

    # Qwen（用于 short/long 生成）
    p.add_argument("--qwen_dir", type=str,
                   default="/mnt/parscratch/users/acs24jw/llmmodels/Qwen2.5-7B-Instruct/Qwen2.5-7B-Instruct",
                   help="Qwen Instruct 模型的本地权重目录（需要包含模型权重与 tokenizer 文件）")
    p.add_argument("--enable_llm", action="store_true",
                   help="启用后 short_simple/long_simple 走 LLM 真实生成；否则回退池采样。")

    # 策略 & 模式
    p.add_argument("--strategies", nargs="+",
                   default=["alvin","short_simple","long_simple","difficulty","misclassified"])
    p.add_argument("--modes", nargs="+",
                   default=["uncontrolled","controlled-balanced","controlled-optimal"])

    # 训练 / AL（尽量小）
    p.add_argument("--rounds", type=int, default=2)
    p.add_argument("--budget", type=int, default=12)
    p.add_argument("--batch_size", type=int, default=24)
    p.add_argument("--max_len", type=int, default=192)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--patience", type=int, default=1)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--seed", type=int, default=42)

    # 覆盖度（极简：rho=0）
    p.add_argument("--fast_coverage", action="store_true", default=True,
                   help="置 True 时跳过几何覆盖计算，rho 指标写 0。")
    return p

if __name__ == "__main__":
    args = build_argparser().parse_args()
    run(args)

