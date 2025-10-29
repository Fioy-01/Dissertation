# -*- coding: utf-8 -*-
"""
bert_real_runner.py
统一运行脚本：池采样（random/entropy/alvin）+ 诊断（difficulty/misclassified）+ 生成（short_simple/long_simple）

主要功能：
- 4轮主动学习，每轮预算B=150，每轮重新训练10个不同种子的模型（from scratch fine-tuning）
- 选样阶段采用10模型概率平均（committee）→ 熵/策略打分 → 选Top-B
- 评估阶段输出 mean±std 与 best-seed 两套结果
- 模式：uncontrolled（不控类） / controlled-balanced（均衡配额） / controlled-optimal（最优配额，源自B.2/B.3拟合与贪心）
- 记录：少数类提升、采样分布π_t、覆盖率κ_t/几何覆盖ρ_t、逐类错率拟合、最优配额建议、收敛速度指标等
- 诊断：difficulty / misclassified 在每轮后对验证集做分析（不并入训练）
- 生成：short_simple / long_simple 可纯生成赛道；或 --gen_mix G 表示每轮先生成G条再从池抽B-G条

依赖你的项目内模块：datasets.py, models.py, train_utils.py, sampling.py, llm_qwen.py
"""

import os
import json
import argparse
import math
import random
import time
from copy import deepcopy
from typing import List, Dict, Tuple, Optional

import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix

import torch

from datasets import load_jsonl, save_jsonl, NliDataset, LABEL2ID, ID2LABEL
from models import Classifier
from train_utils import set_seed, build_loader, train_with_early_stop
from sampling import select as al_select
from llm_qwen import QwenJudge
from datetime import datetime


# ----------------------------
# Utilities
# ----------------------------

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
    "kappa_uniform","kappa_invfreq","rho_t_max","rho_t_p95",
    "gen_G_used","gen_pass","gen_pass_rate"
]

def log_round_metrics(round_log_path, data_dict):
    """写入一轮 AL 的结果到 CSV，如果文件不存在则先写表头"""
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
            data_dict.get("gen_G_used", 0),
            data_dict.get("gen_pass", 0),
            f"{data_dict.get('gen_pass_rate', 0.0):.4f}"
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


def build_dataset_and_loader(rows: List[Dict], tokenizer, batch_size: int, max_len: int, shuffle: bool, has_label_maybe=True):
    ds = NliDataset(rows, tokenizer=tokenizer, max_len=max_len)
    return ds, build_loader(ds, batch_size=batch_size, shuffle=shuffle, tokenizer=tokenizer, max_len=max_len)


def compute_per_class_metrics(golds: List[int], preds: List[int], n_classes: int = 3):
    acc = accuracy_score(golds, preds)
    # per-class metrics
    prec, rec, f1, support = precision_recall_fscore_support(golds, preds, labels=list(range(n_classes)), average=None, zero_division=0)
    # macro
    macro_prec = float(np.mean(prec))
    macro_rec = float(np.mean(rec))
    macro_f1 = float(np.mean(f1))
    cm = confusion_matrix(golds, preds, labels=list(range(n_classes)))
    return {
        "accuracy": float(acc),
        "macro_precision": macro_prec,
        "macro_recall": macro_rec,
        "macro_f1": macro_f1,
        "per_class_precision": prec.tolist(),
        "per_class_recall": rec.tolist(),
        "per_class_f1": f1.tolist(),
        "support": support.tolist(),
        "confusion_matrix": cm.tolist(),
    }


def eval_model_on_split(clf: Classifier, rows: List[Dict], batch_size: int, max_len: int):
    # requires labels present
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


class EnsembleModel:
    """Wrap multiple trained Classifier to provide averaged predict_proba and a stable encode."""
    def __init__(self, members: List[Classifier]):
        assert len(members) > 0
        self.members = members
        self.tokenizer = members[0].get_tokenizer()
        self.device = members[0].device

    def predict_proba(self, dataloader) -> np.ndarray:
        probs_list = []
        for m in self.members:
            p = m.predict_proba(dataloader)  # [N, C]
            probs_list.append(p)
        return np.mean(np.stack(probs_list, axis=0), axis=0)

    def encode(self, dataloader) -> np.ndarray:
        # Use member[0] for deterministic embeddings to save compute.
        return self.members[0].encode(dataloader, normalize=False)

    def get_tokenizer(self):
        return self.tokenizer


def majority_minority_classes(counts: Dict[str, int]) -> Tuple[List[str], List[str]]:
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


def predicted_class_distribution(probs: np.ndarray) -> Dict[str, int]:
    pred = probs.argmax(axis=1)
    d = {"entailment": 0, "neutral": 0, "contradiction": 0}
    for p in pred:
        d[ID2LABEL[int(p)]] += 1
    return d


def apply_balanced_quota_with_predictions(sorted_indices: List[int],
                                          unlabeled_pool: List[Dict],
                                          probs_on_pool: np.ndarray,
                                          B: int,
                                          start_major: str = "entailment") -> List[int]:
    """Use predicted classes to fill per-class quotas as evenly as possible."""
    base = B // 3
    extra = B - base * 3
    order = ["entailment", "neutral", "contradiction"]
    # rotate which class gets the extra 1 each round if needed (caller can adjust)
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
        # backfill from remaining in the original sorted order
        taken = set(selected)
        for idx in sorted_indices:
            if len(selected) >= B:
                break
            if idx not in taken:
                selected.append(idx)
    return selected[:B]


def fit_power_law_errors(n_list: List[int], err_list: List[float]) -> Tuple[float, float, float, float]:
    """
    Fit: err(n) = a * (n + b)^(-gamma) + c
    简化拟合：少回合时固定 gamma=0.5，并用最小二乘粗略估计 a,b,c。
    """
    if len(n_list) < 3:
        # fallback constants
        return 1.0, 1.0, 0.5, min(err_list) if err_list else 0.0
    gamma = 0.5  # shared fixed gamma for stability
    # transform y - c ≈ a * (n+b)^(-gamma). We'll grid-search c,b small ranges to stabilize.
    best = None
    y = np.array(err_list, dtype=np.float64)
    n = np.array(n_list, dtype=np.float64)
    for c in np.linspace(max(0.0, y.min() - 0.05), min(0.3, y.min() + 0.1), 4):
        for b in np.linspace(0.0, max(1.0, n.max() * 0.1), 5):
            X = (n + b) ** (-gamma)
            # linear in a: y-c ≈ a*X
            denom = (X**2).sum() + 1e-9
            a = float(np.sum((y - c) * X) / denom)
            y_hat = a * X + c
            se = float(((y - y_hat) ** 2).mean())
            cand = (se, a, b, gamma, c)
            if (best is None) or (se < best[0]):
                best = cand
    _, a, b, gamma, c = best
    return a, b, gamma, c


def estimate_class_quota_optimal(B: int,
                                 n_k: Dict[str, int],
                                 err_curve_params: Dict[str, Tuple[float, float, float, float]],
                                 eta_k: Dict[str, float]) -> Dict[str, int]:
    """
    根据 B.3 贪心，为每类分配 m_k 以最大化 ΔF1 总收益。
    Δε_k(m) ≈ a * [(n+b)^(-γ) - (n+m+b)^(-γ)]
    ΔF1_k ≈ η_k * Δε_k
    贪心：每次把1个名额分给当前边际收益最大的类。
    """
    classes = ["entailment", "neutral", "contradiction"]
    m = {k: 0 for k in classes}

    def marginal(k, add_m: int = 1):
        a, b, gamma, c = err_curve_params.get(k, (1.0, 1.0, 0.5, 0.0))
        n0 = n_k.get(k, 0) + m[k]
        # Δε = a * ( (n0+b)^(-γ) - (n0+add_m+b)^(-γ) )
        d_eps = a * ((n0 + b) ** (-gamma) - (n0 + add_m + b) ** (-gamma))
        return eta_k.get(k, 1.0) * d_eps

    for _ in range(B):
        gains = {k: marginal(k, 1) for k in classes}
        pick = max(gains.items(), key=lambda x: x[1])[0]
        m[pick] += 1
    return m


def compute_coverage_metrics(labeled_rows: List[Dict], unlabeled_rows: List[Dict], ref_model: Classifier, batch_size: int, max_len: int):
    """Compute rho_t (max and p95) using embeddings distances from U to nearest in S."""
    tok = ref_model.get_tokenizer()
    # encode labeled
    ds_L, ld_L = build_dataset_and_loader(labeled_rows, tok, batch_size=batch_size, max_len=max_len, shuffle=False)
    emb_L = ref_model.encode(ld_L, normalize=False)  # [|S|, D]
    if len(unlabeled_rows) == 0 or emb_L.shape[0] == 0:
        return {"rho_t_max": 0.0, "rho_t_p95": 0.0}
    # encode a sample of U to save time if huge
    sample_U = unlabeled_rows
    if len(sample_U) > 5000:
        sample_idx = np.random.choice(len(sample_U), size=5000, replace=False).tolist()
        sample_U = [unlabeled_rows[i] for i in sample_idx]
    ds_U, ld_U = build_dataset_and_loader(sample_U, tok, batch_size=batch_size, max_len=max_len, shuffle=False)
    emb_U = ref_model.encode(ld_U, normalize=False)  # [|U'|, D]
    # compute min distance to S for each x in U'
    # Use chunked computation to control memory
    S = emb_L.astype(np.float32)
    U = emb_U.astype(np.float32)
    S_sq = (S ** 2).sum(axis=1)
    dmins = []
    chunk = 1024
    for i in range(0, U.shape[0], chunk):
        Uc = U[i:i+chunk]
        U_sq = (Uc ** 2).sum(axis=1, keepdims=True)  # [c,1]
        dot = Uc @ S.T  # [c, |S|]
        d2 = U_sq + S_sq[None, :] - 2.0 * dot
        dmin = np.sqrt(np.maximum(0.0, d2.min(axis=1)))
        dmins.append(dmin)
    dmins = np.concatenate(dmins, axis=0)
    rho_max = float(np.max(dmins))
    rho_p95 = float(np.percentile(dmins, 95))
    return {"rho_t_max": rho_max, "rho_t_p95": rho_p95}


# ----------------------------
# Main Runner
# ----------------------------

def run(args):
    # Paths
    INIT_DIR = args.init_dir
    VAL_PATH = args.val_path
    TEST_PATH = args.test_path
    MODEL_DIR = args.model_dir
    QWEN_DIR = args.qwen_dir if args.qwen_dir else None

    # Output organization
    base_out = args.output_root
    ensure_dir(base_out)

    # Load splits
    init_L, init_U = load_split(INIT_DIR)
    val_rows, test_rows = load_eval_sets(VAL_PATH, TEST_PATH)

    # Static tokenizer (from a temp classifier) for loaders; real models are created per seed anyway
    temp_clf = Classifier(MODEL_DIR, num_labels=3, fp16=True)
    tokenizer = temp_clf.get_tokenizer()
    del temp_clf

    # LLM client (optional)
    llm_client = None
    if args.enable_llm and QWEN_DIR is not None:
        llm_client = QwenJudge(QWEN_DIR, fp16=True)

    # Strategies to run
    strategies = [s.strip() for s in args.strategies.split(",") if s.strip()]
    strategies = [strat_name_clean(s) for s in strategies]

    # For controlled-balanced rotation
    balance_rotation = ["entailment", "neutral", "contradiction"]
    balance_rot_ptr = 0

    # Global logs collector for summary
    summary_rows = []

    for strat in strategies:
        for mode in args.modes:  # e.g., ["uncontrolled","controlled-balanced","controlled-optimal"]
            mode = mode.lower()
            # Skip invalid combos for analysis/gen-only
            if strat in ("difficulty", "misclassified") and mode != "uncontrolled":
                pass

            exp_name = f"BERT_ratio-real_{strat}_{mode}"

            # ---- Adaptive-G state (only for pure generation strategies) ----
            is_gen_strategy = strat in ("short_simple", "long_simple")
            G_current = int(args.gen_mix) if is_gen_strategy else 0
            prev_macro_f1_mean = None      # 基线：上一轮验证 Macro-F1（mean）
            rolling_gen_gain = None        # 逐轮 EMA 的“每样本增益”（ΔF1_gen per sample）
            alpha_ma = 0.5                 # EMA 系数

            out_dir = os.path.join(base_out, exp_name)
            ensure_dir(out_dir)
            ensure_dir(os.path.join(out_dir, "plots"))
            with open(os.path.join(out_dir, "run_config.json"), "w", encoding="utf-8") as f:
                json.dump({
                    "timestamp": timestamp(),
                    "strategy": strat,
                    "mode": mode,
                    "init_dir": INIT_DIR,
                    "val_path": VAL_PATH,
                    "test_path": TEST_PATH,
                    "model_dir": MODEL_DIR,
                    "qwen_dir": QWEN_DIR if args.enable_llm else None,
                    "rounds": args.rounds,
                    "budget_per_round": args.budget,
                    "ensemble_size": args.ensemble_size,
                    "batch_size": args.batch_size,
                    "max_len": args.max_len,
                    "epochs": args.epochs,
                    "patience": args.patience,
                    "lr": args.lr,
                    "gen_mix": args.gen_mix,
                }, f, ensure_ascii=False, indent=2)

            # Initialize per-exp state
            L = deepcopy(init_L)  # labeled set
            U = deepcopy(init_U)  # unlabeled pool
            # Book-keeping for learning curve fitting
            per_round_stats = []
            # For optimal quota inputs
            eta_k = {k: 1.0 for k in ["entailment", "neutral", "contradiction"]}  # sensitivity init

            for t in range(1, args.rounds + 1):
                print(f"\n=== [{exp_name}] Round {t}/{args.rounds} ===")
                round_log_path = os.path.join(out_dir, "round_log.csv")
                new_csv = not os.path.exists(round_log_path)

                # -------------------
                # (A) Train ensemble (10 seeds) FROM SCRATCH on current L
                # -------------------
                members: List[Classifier] = []
                seed_list = [args.seed + i for i in range(args.ensemble_size)]
                for s in seed_list:
                    set_seed(s)
                    clf = Classifier(MODEL_DIR, num_labels=3, fp16=True)
                    tok_local = clf.get_tokenizer()
                    _, train_loader = build_dataset_and_loader(L, tok_local, batch_size=args.batch_size, max_len=args.max_len, shuffle=True)
                    _, val_loader = build_dataset_and_loader(val_rows, tok_local, batch_size=args.batch_size, max_len=args.max_len, shuffle=False)
                    train_with_early_stop(
                        clf, train_loader, val_loader,
                        epochs=args.epochs, patience=args.patience, lr=args.lr,
                        total_steps_hint=len(train_loader) * args.epochs
                    )
                    members.append(clf)

                # Evaluate each member on val to compute mean±std and identify best-seed
                member_metrics = []
                for i, clf in enumerate(members):
                    m = eval_model_on_split(clf, val_rows, batch_size=args.batch_size, max_len=args.max_len)
                    m["seed"] = seed_list[i]
                    member_metrics.append(m)
                macro_f1s = [m["macro_f1"] for m in member_metrics]
                best_idx = int(np.argmax(macro_f1s))
                best_clf = members[best_idx]
                best_seed = seed_list[best_idx]

                # === 关键修复：构建 ensemble，并统一 tokenizer ===
                ensemble = EnsembleModel(members)
                tok = ensemble.get_tokenizer()

                # Aggregate mean±std
                def meanstd(key):
                    arr = np.array([m[key] for m in member_metrics], dtype=np.float32)
                    return float(arr.mean()), float(arr.std())

                macro_f1_mean, macro_f1_std = meanstd("macro_f1")
                acc_mean, acc_std = meanstd("accuracy")
                macro_prec_mean, _ = meanstd("macro_precision")
                macro_rec_mean, _ = meanstd("macro_recall")

                # Per-class mean metrics (for minority highlight)
                per_class_f1_mean = np.mean(np.stack([m["per_class_f1"] for m in member_metrics], axis=0), axis=0).tolist()
                per_class_rec_mean = np.mean(np.stack([m["per_class_recall"] for m in member_metrics], axis=0), axis=0).tolist()
                per_class_prec_mean = np.mean(np.stack([m["per_class_precision"] for m in member_metrics], axis=0), axis=0).tolist()

                # Determine minority class at start of this round (by L's true counts)
                cnt_L = count_by_label(L)
                mins, maxs = majority_minority_classes(cnt_L)
                # For logging, if multiple minorities, pick the first deterministically
                minority = mins[0] if mins else "neutral"
                min_idx = LABEL2ID[minority]
                minority_f1 = float(per_class_f1_mean[min_idx])
                minority_rec = float(per_class_rec_mean[min_idx])
                minority_prec = float(per_class_prec_mean[min_idx])

                # -------------------
                # (B) Selection phase
                # -------------------
                B = args.budget

                # Generation quota for this round
                if is_gen_strategy:
                    # 纯生成策略：使用当前 G（可自适应）
                    gen_G = max(0, int(G_current))
                else:
                    # 非生成策略：不启用生成与自适应
                    gen_G = 0

                # === 关键修复：初始化，避免 rest_B==0 时 selected_indices 未定义 ===
                selected_indices: List[int] = []

                # (B1) Generation part (only if gen_G>0 and LLM enabled)
                gen_samples: List[Dict] = []
                gen_attempts = 0
                if gen_G > 0 and args.enable_llm and llm_client is not None:
                    # 对纯生成策略：用 strat 自己。保留 gen_mode 以兼容两种生成器
                    sname = "short_simple" if (strat == "short_simple") else \
                            "long_simple"  if (strat == "long_simple")  else args.gen_mode
                    _idxs, new_samples = al_select(
                        strategy=sname,
                        unlabeled_pool=[],
                        model=None, tokenizer=None,
                        batch_size=args.batch_size, max_len=args.max_len,
                        budget_b=gen_G, seed=args.seed,
                        llm_client=llm_client, n_votes=8, balance_generated=True, llm_max_trials=2000
                    )
                    if new_samples:
                        gen_samples.extend(new_samples)
                    # 生成接口不返回尝试次数；用“申请配额”近似尝试量
                    gen_attempts = max(1, gen_G)

                gen_pass = len(gen_samples)
                approx_pass_rate = (gen_pass / float(gen_attempts)) if gen_attempts > 0 else 0.0

                # (B2) Pool selection part
                rest_B = B - len(gen_samples)
                if rest_B > 0:
                    # Uncontrolled first to get a scored order; controlled uses predicted labels to re-balance.
                    if strat in ("random", "entropy", "alvin"):
                        idxs, new_samples_unused = al_select(
                            strategy=strat,
                            unlabeled_pool=U,
                            model=ensemble,
                            tokenizer=tok,
                            batch_size=args.batch_size,
                            max_len=args.max_len,
                            budget_b=rest_B if mode == "uncontrolled" else len(U),  # for controlled we take a longer candidate list
                            seed=args.seed,
                            # ALVIN extras
                            labeled_set=L,
                            dynamics=None,
                            low_conf_thresh=0.5,
                            anchors_per_class=20,
                            alpha=2.0,
                            knn_k=15,
                            rng_seed=args.seed,
                            normalize_encodings=False,
                            use_faiss=False,
                            faiss_use_gpu=False,
                        )

                        if mode == "uncontrolled":
                            selected_indices = idxs[:rest_B]
                        else:
                            # Need predicted labels on U for quota
                            dsU = NliDataset(U, tokenizer=tok, max_len=args.max_len)
                            ldU = build_loader(dsU, batch_size=args.batch_size, shuffle=False, tokenizer=tok, max_len=args.max_len)
                            pool_probs = ensemble.predict_proba(ldU)  # [|U|,3]

                            if mode == "controlled-balanced":
                                start_major = balance_rotation[balance_rot_ptr % 3]
                                balance_rot_ptr += 1
                                # idxs is a ranking by strategy; apply balanced-quota on this ranking
                                selected_indices = apply_balanced_quota_with_predictions(
                                    sorted_indices=idxs, unlabeled_pool=U, probs_on_pool=pool_probs, B=rest_B, start_major=start_major
                                )
                            elif mode == "controlled-optimal":
                                # Estimate optimal quota using B.2/B.3 based on past rounds statistics
                                # Prepare inputs: current per-class labeled counts n_k
                                n_k = count_by_label(L)
                                # Build simple error curves from historical per-round stats
                                err_params = {}
                                for k_name in ["entailment", "neutral", "contradiction"]:
                                    # Gather n_t(k) and class error at each past round (1 - recall_k)
                                    n_list = []
                                    err_list = []
                                    for rs in per_round_stats:
                                        n_list.append(rs["n_counts"][k_name])
                                        # 1 - recall_k from per-class mean on that round
                                        err_list.append(1.0 - rs["per_class_recall_mean"][LABEL2ID[k_name]])
                                    a, b, gamma, c = fit_power_law_errors(n_list, err_list) if len(n_list) >= 2 else (1.0, 1.0, 0.5, 0.1)
                                    err_params[k_name] = (a, b, gamma, c)
                                # Update eta_k using a smoothed mapping Macro-F1 sensitivity; here use a soft heuristic ~ class weight
                                # If class recall was low, give larger eta
                                for k in ["entailment", "neutral", "contradiction"]:
                                    last_rec = per_class_rec_mean[LABEL2ID[k]]
                                    eta_k[k] = float(1.0 + max(0.0, (0.7 - last_rec)))  # simple smoothing

                                quota = estimate_class_quota_optimal(rest_B, n_k, err_params, eta_k)  # dict lab->m
                                # Build buckets by predicted label from the ranked list
                                y_pred = pool_probs.argmax(axis=1)
                                buckets = {"entailment": [], "neutral": [], "contradiction": []}
                                chosen = []
                                for idx in idxs:
                                    lab = ID2LABEL[int(y_pred[idx])]
                                    buckets[lab].append(idx)
                                for lab, need in quota.items():
                                    chosen.extend(buckets[lab][:need])
                                # Backfill if shortage
                                if len(chosen) < rest_B:
                                    taken = set(chosen)
                                    for idx in idxs:
                                        if len(chosen) >= rest_B:
                                            break
                                        if idx not in taken:
                                            chosen.append(idx)
                                selected_indices = chosen[:rest_B]
                            else:
                                selected_indices = idxs[:rest_B]

                    elif strat in ("short_simple", "long_simple"):
                        # Pure generation, no pool usage in this branch
                        idxs, new_samples = al_select(
                            strategy=strat,
                            unlabeled_pool=[],
                            model=None, tokenizer=None,
                            batch_size=args.batch_size, max_len=args.max_len,
                            budget_b=rest_B, seed=args.seed,
                            llm_client=llm_client, n_votes=8, balance_generated=True, llm_max_trials=2000
                        )
                        gen_samples.extend(new_samples or [])

                    elif strat in ("difficulty", "misclassified"):
                        # Diagnostic only: on validation set
                        idxs, _ = al_select(
                            strategy=strat,
                            unlabeled_pool=[],
                            model=ensemble, tokenizer=tok,
                            batch_size=args.batch_size, max_len=args.max_len,
                            budget_b=rest_B, seed=args.seed,
                            candidate_labeled_set=val_rows,
                            per_class_k=max(1, rest_B // 3),
                            llm_client=llm_client
                        )
                        # Save diagnostics; do not change L/U
                        diag_path = os.path.join(out_dir, f"diagnostic_{strat}_round{t}.json")
                        with open(diag_path, "w", encoding="utf-8") as f:
                            json.dump({"indices": idxs}, f, ensure_ascii=False, indent=2)
                        selected_indices = []
                    else:
                        raise ValueError(f"Unknown strategy: {strat}")

                # Save selected & generated artifacts
                if selected_indices:
                    with open(os.path.join(out_dir, f"selected_indices_round{t}.json"), "w", encoding="utf-8") as f:
                        json.dump(selected_indices, f, ensure_ascii=False, indent=2)
                    sel_rows = [U[i] for i in selected_indices]
                    save_jsonl(os.path.join(out_dir, f"selected_samples_round{t}.jsonl"), sel_rows)

                if gen_samples:
                    save_jsonl(os.path.join(out_dir, f"generated_samples_round{t}.jsonl"), gen_samples)

                # -------------------
                # (C) Update L/U
                # -------------------
                added_rows = []
                if selected_indices:
                    sel_set = set(selected_indices)
                    remain = []
                    for i, r in enumerate(U):
                        (added_rows if i in sel_set else remain).append(r)
                    U = remain
                if gen_samples:
                    added_rows.extend(gen_samples)
                # Append to L
                if added_rows:
                    # ensure labels are ints for NliDataset
                    for r in added_rows:
                        if isinstance(r.get("label", None), str):
                            r["label"] = LABEL2ID[r["label"]]
                    L.extend(added_rows)

                # -------------------
                # (D) Coverage metrics (ρ_t) & kappa-like counts coverage
                # -------------------
                cov = compute_coverage_metrics(L, U, best_clf, batch_size=args.batch_size, max_len=args.max_len)
                # Count-coverage (kappa): uniform & inverse-frequency weighted
                n_counts = count_by_label(L)  # true labels in L
                N_total_per_class = count_by_label(init_L + init_U)  # dataset-level totals (simulator)
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

                # -------------------
                # (E) Persist round log (mean±std on VAL)
                # -------------------
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
                    "macro_f1": macro_f1_mean,
                    "accuracy": acc_mean,
                    "macro_precision": macro_prec_mean,
                    "macro_recall": macro_rec_mean,
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
                    "gen_G_used": gen_G,
                    "gen_pass": gen_pass,
                    "gen_pass_rate": approx_pass_rate
                })

                # （可选）为最优配额拟合保留历史
                per_round_stats.append({
                    "n_counts": n_counts,
                    "per_class_recall_mean": per_class_rec_mean,
                })

                # -------------------
                # (E2) Adaptive-G update (only for pure generation strategies)
                # -------------------
                if is_gen_strategy and args.gen_adaptive:
                    # 使用“当前轮宏F1均值”与“上一轮宏F1均值”做差分；若上一轮为空则只走通过率降档
                    if prev_macro_f1_mean is not None and gen_pass > 0:
                        delta_f1_gen_per_sample = (macro_f1_mean - prev_macro_f1_mean) / float(gen_pass)

                        # 若还没有滚动基准，则初始化
                        if rolling_gen_gain is None:
                            rolling_gen_gain = delta_f1_gen_per_sample

                        up_th = rolling_gen_gain * (1.0 + args.gen_epsilon)
                        down_th = rolling_gen_gain * (1.0 - args.gen_epsilon)

                        G_new = G_current
                        if delta_f1_gen_per_sample >= up_th:
                            G_new = min(args.gen_max, G_current + args.gen_step)
                        elif delta_f1_gen_per_sample <= down_th:
                            G_new = max(args.gen_min, G_current - args.gen_step)

                        # 通过率保护：过低则强制下调一步
                        if approx_pass_rate < args.gen_pass_rate_thresh:
                            G_new = max(args.gen_min, G_new - args.gen_step)

                        # EMA 更新滚动基线
                        rolling_gen_gain = alpha_ma * delta_f1_gen_per_sample + (1 - alpha_ma) * rolling_gen_gain
                        G_current = int(G_new)

                        # 追加一行“内联注释”到 round_log，方便 grep
                        with open(round_log_path, "a", encoding="utf-8") as fw:
                            fw.write(f"#GEN_ADAPT,round={t},G_before={int(max(0, G_current))},"
                                     f"delta_f1_gen_per_sample={delta_f1_gen_per_sample:.6f},"
                                     f"rolling_gen_gain={rolling_gen_gain:.6f},"
                                     f"pass_rate={approx_pass_rate:.4f},"
                                     f"G_after={int(G_current)}\n")
                    else:
                        # 没有上一轮作为对比 / 本轮无通过样本：仅根据低通过率下调
                        if approx_pass_rate < args.gen_pass_rate_thresh:
                            G_current = int(max(args.gen_min, G_current - args.gen_step))
                            with open(round_log_path, "a", encoding="utf-8") as fw:
                                fw.write(f"#GEN_ADAPT,round={t},G_only_passrate_down,"
                                         f"pass_rate={approx_pass_rate:.4f},G_after={int(G_current)}\n")

                    # 更新下一轮的基线 F1
                    prev_macro_f1_mean = macro_f1_mean
                else:
                    # 非生成策略：保持基线更新但不做自适应（可省略这行）
                    prev_macro_f1_mean = macro_f1_mean

                # -------------------
                # (F) Diagnostics: difficulty/misclassified on VAL
                # -------------------
                if args.enable_llm and llm_client is not None:
                    if args.run_diagnostics:
                        for dstrat in ["difficulty", "misclassified"]:
                            try:
                                d_idxs, _ = al_select(
                                    strategy=dstrat,
                                    unlabeled_pool=[],
                                    model=ensemble, tokenizer=tok,
                                    batch_size=args.batch_size, max_len=args.max_len,
                                    budget_b=max(3, B // 10),
                                    seed=args.seed,
                                    candidate_labeled_set=val_rows,
                                    per_class_k=max(1, B // 30),
                                    llm_client=llm_client
                                )
                                with open(os.path.join(out_dir, f"diagnostic_{dstrat}_round{t}.json"), "w", encoding="utf-8") as f:
                                    json.dump({"indices": d_idxs}, f, ensure_ascii=False, indent=2)
                            except Exception as e:
                                print(f"[Warn] Diagnostic {dstrat} failed: {e}")

                # -------------------
                # (G) Save per-round best checkpoint (optional small)
                # -------------------
                # 这里为了节约空间，不保存权重，只保存best-seed编号与验证指标
                with open(os.path.join(out_dir, f"best_seed_round{t}.json"), "w", encoding="utf-8") as f:
                    json.dump({"best_seed": best_seed, "macro_f1": macro_f1s[best_idx]}, f, ensure_ascii=False, indent=2)

            # End of rounds
            # add summary row (final macro_f1 on VAL mean ± std from last round)
            summary_rows.append({
                "exp": exp_name,
                "strategy": strat,
                "mode": mode,
                "rounds": args.rounds,
                "B": args.budget,
            })

    # Write summary index
    with open(os.path.join(base_out, "SUMMARY_INDEX.json"), "w", encoding="utf-8") as f:
        json.dump(summary_rows, f, ensure_ascii=False, indent=2)

    print("\nAll experiments completed. Outputs saved to:", base_out)


def build_argparser():
    p = argparse.ArgumentParser(description="Unified runner for AL on FEVER ratio_real with BERT base.")
    # Paths
    p.add_argument("--init_dir", type=str,
                   default="/users/acs24jw/nli_project/mis_fever/data/fever_2000/ratio_real")
    p.add_argument("--val_path", type=str,
                   default="/users/acs24jw/nli_project/mis_fever/data/dev_f.jsonl")
    p.add_argument("--test_path", type=str,
                   default="/users/acs24jw/nli_project/mis_fever/data/test_f.jsonl")
    p.add_argument("--model_dir", type=str,
                   default="/mnt/parscratch/users/acs24jw/models/bert-base-uncased_model")
    p.add_argument("--qwen_dir", type=str,
                   default="/mnt/parscratch/users/acs24jw/llmmodels/Qwen2.5-7B-Instruct/Qwen2.5-7B-Instruct")
    p.add_argument("--output_root", type=str,
                   default="/users/acs24jw/nli_project/mis_fever/outputs")

    # Strategies & modes
    p.add_argument("--strategies", type=str,
                   default="random,entropy,alvin,short_simple,long_simple,difficulty,misclassified")
    p.add_argument("--modes", nargs="+",
                   default=["uncontrolled", "controlled-balanced", "controlled-optimal"],
                   help="Which control modes to run; diagnostics ignore this.")
    p.add_argument("--gen_mode", type=str, choices=["short_simple", "long_simple"], default="short_simple",
                   help="When --gen_mix>0 or strategy is generation, which generator to use.")
    p.add_argument("--gen_mix", type=int, default=0,
                   help="If >0 and strategy in pool class, generate G items first then pick B-G from pool.")

    # Training/AL config
    p.add_argument("--rounds", type=int, default=4)
    p.add_argument("--budget", type=int, default=150)
    p.add_argument("--ensemble_size", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--max_len", type=int, default=256)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--patience", type=int, default=2)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--seed", type=int, default=42)

    # LLM / diagnostics
    p.add_argument("--enable_llm", action="store_true", help="Enable Qwen-based generation/difficulty.")
    p.add_argument("--run_diagnostics", action="store_true", help="Run difficulty/misclassified diagnostics on VAL each round.")

    # ---- Adaptive generation (only for generation strategies) ----
    p.add_argument("--gen_adaptive", action="store_true",
                   help="Enable adaptive G only for pure generation strategies (short_simple/long_simple).")
    p.add_argument("--gen_step", type=int, default=15, help="Step size when adjusting G.")
    p.add_argument("--gen_min", type=int, default=15, help="Lower bound for G.")
    p.add_argument("--gen_max", type=int, default=60, help="Upper bound for G.")
    p.add_argument("--gen_epsilon", type=float, default=0.05,
                   help="Increase G if ΔF1_gen ≥ rolling*(1+ε); decrease if ≤ *(1-ε).")
    p.add_argument("--gen_pass_rate_thresh", type=float, default=0.15,
                   help="If generation pass-rate below this, decrease G by one step.")

    return p


if __name__ == "__main__":
    args = build_argparser().parse_args()
    run(args)

