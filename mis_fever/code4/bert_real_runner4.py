# -*- coding: utf-8 -*-
import os
import csv
import json
import time
import argparse
from datetime import datetime
from typing import List, Dict, Any, Optional

import numpy as np
import torch

from datasets import NliDataset
from train_utils import build_loader, set_seed, train_with_early_stop
from models import Classifier
from vagery import (
    Rectifier,
    VariationalHead,
    VaGeRyLossCfg,
    AdaptiveCfg,
    clone_teacher_from_student,
    memorization_phase,
    reception_phase,
    RectifiedModel,
)
import sampling

# ----------------------------
# JSONL helpers
# ----------------------------

def read_jsonl(path: str) -> List[Dict[str, Any]]:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    return data


def write_jsonl(path: str, rows: List[Dict[str, Any]]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

# ----------------------------
# Utils
# ----------------------------
LABELS = ["entailment", "neutral", "contradiction"]
L2I = {l: i for i, l in enumerate(LABELS)}
I2L = {i: l for l, i in L2I.items()}


# 【新增】从 runner3 引入的工具函数
def majority_minority_classes(counts: Dict[str, int]) -> (List[str], List[str]):
    if not counts:
        return [], []
    min_cnt = min(counts.values())
    max_cnt = max(counts.values())
    return [k for k, v in counts.items() if v == min_cnt], [k for k, v in counts.items() if v == max_cnt]


def count_by_label(rows: List[Dict[str, Any]]) -> Dict[str, int]:
    cnt = {l: 0 for l in LABELS}
    for r in rows:
        label = r.get("label")
        if isinstance(label, int):
            label = I2L.get(label)
        if label in cnt:
            cnt[label] += 1
    return cnt


def scarcity_pi(counts: Dict[str, int]) -> np.ndarray:
    arr = np.array([counts.get(l, 0) for l in LABELS], dtype=np.float32)
    m = float(max(arr.max(), 1.0))
    return arr / m


def timestamp_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def ensure_dir(p: str):
    if p and not os.path.exists(p):
        os.makedirs(p, exist_ok=True)

# ----------------------------
# Coverage metrics (简版)
# ----------------------------

def compute_coverage_metrics(selected_rows: List[Dict[str, Any]], unlabeled_rows: List[Dict[str, Any]],
                             enc_model: Classifier, tokenizer, batch_size: int, max_len: int) -> Dict[str, float]:
    c = count_by_label(selected_rows)
    arr = np.array([c["entailment"], c["neutral"], c["contradiction"]], dtype=np.float32)
    if arr.sum() <= 0:
        kappa_uniform = 0.0; kappa_invfreq = 0.0
    else:
        p = arr / arr.sum()
        gini = np.abs(p.reshape(-1, 1) - p.reshape(1, -1)).sum() / (2.0 * len(p))
        kappa_uniform = float(1.0 - gini)
        w = np.ones_like(p); w[arr == arr.min()] = 2.0
        pw = p * w; pw = pw / max(pw.sum(), 1e-8)
        gini_w = np.abs(pw.reshape(-1, 1) - pw.reshape(1, -1)).sum() / (2.0 * len(pw))
        kappa_invfreq = float(1.0 - gini_w)

    try:
        dsS = NliDataset(selected_rows, tokenizer=tokenizer, max_len=max_len)
        ldS = build_loader(dsS, batch_size=batch_size, shuffle=False, tokenizer=tokenizer, max_len=max_len)
        S = enc_model.encode(ldS)
        dsU = NliDataset(unlabeled_rows, tokenizer=tokenizer, max_len=max_len)
        ldU = build_loader(dsU, batch_size=batch_size, shuffle=False, tokenizer=tokenizer, max_len=max_len)
        U = enc_model.encode(ldU)
        from numpy.linalg import norm
        dmins = []
        for s in S:
            d = norm(U - s, axis=1).min()
            dmins.append(d)
        arrd = np.array(dmins, dtype=np.float32)
        rho_max = float(arrd.max()) if arrd.size else 0.0
        rho_p95 = float(np.percentile(arrd, 95)) if arrd.size else 0.0
    except Exception:
        rho_max = 0.0; rho_p95 = 0.0

    return {
        "kappa_uniform": kappa_uniform,
        "kappa_invfreq": kappa_invfreq,
        "rho_t_max": rho_max,
        "rho_t_p95": rho_p95,
    }

# ----------------------------
# 【修改】 Args
# ----------------------------

def parse_args():
    ap = argparse.ArgumentParser("Enhanced VaGeRy Runner with Detailed Metrics")
    # ... (参数基本保持不变，新增 mode)
    ap.add_argument("--init_labeled", type=str, required=True)
    ap.add_argument("--unlabeled_pool", type=str, required=True)
    ap.add_argument("--val_path", type=str, required=True)
    ap.add_argument("--test_path", type=str, default=None)
    ap.add_argument("--model_dir", type=str, required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--patience", type=int, default=2)
    ap.add_argument("--ema_m", type=float, default=0.99)
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--budget", type=int, default=150)
    ap.add_argument("--strategy", type=str, default="alvin_var_cls",
                    choices=["alvin_var_cls", "alvin", "entropy", "random"])
    # 【新增】mode 参数
    ap.add_argument("--mode", type=str, default="uncontrolled", help="Selection mode, for logging purposes.")
    ap.add_argument("--eta_schedule", type=str, default="0.6,0.6,0.8,0.9")
    ap.add_argument("--floor_rho", type=float, default=0.5)
    ap.add_argument("--cap_frac", type=float, default=0.4)
    ap.add_argument("--mc_passes", type=int, default=3)
    ap.add_argument("--boundary_tau", type=float, default=0.1)
    ap.add_argument("--residual_boost_frac", type=float, default=0.15)
    # 去偏/自适应
    ap.add_argument("--warmup_rounds", type=int, default=1)
    ap.add_argument("--eps_u0", type=float, default=0.06)
    ap.add_argument("--eps_f0", type=float, default=0.03)
    ap.add_argument("--alpha_u", type=float, default=0.5)
    ap.add_argument("--alpha_f", type=float, default=0.5)
    ap.add_argument("--beta_u", type=float, default=0.5)
    ap.add_argument("--beta_f", type=float, default=0.5)
    ap.add_argument("--gamma_v", type=float, default=1.0)
    ap.add_argument("--lambda_cls", type=float, default=0.5)
    # 训练端稳健
    ap.add_argument("--la_tau", type=float, default=0.7)
    ap.add_argument("--focal_gamma", type=float, default=1.5)
    # 伪标/一致性（节制）
    ap.add_argument("--use_pseudo_label", action="store_true")
    ap.add_argument("--pseudo_label_max_p", type=float, default=0.9)
    ap.add_argument("--pseudo_label_max_ratio", type=float, default=0.2, help="每轮伪标数量占B的上限比例")
    ap.add_argument("--pseudo_label_weight", type=float, default=0.5)
    ap.add_argument("--pseudo_label_max_var", type=float, default=0.3, help="过滤高方差样本")
    # 输出
    ap.add_argument("--output_dir", type=str, required=True)
    ap.add_argument("--run_name", type=str, default=f"vagery_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    return ap.parse_args()

# ----------------------------
# 【修改】 Runner 主体
# ----------------------------

def main():
    args = parse_args()
    set_seed(args.seed)
    
    run_output_dir = os.path.join(args.output_dir, args.run_name)
    ensure_dir(run_output_dir)
    print(f"[*] All outputs will be saved to: {run_output_dir}")
    
    # 【修改】使用 runner3 的完整表头
    ROUND_LOG_HEADER = [
        "timestamp","strategy","mode","round","labeled_total","B",
        "minority_class","minority_f1","minority_recall","minority_precision",
        "macro_f1","accuracy","macro_precision","macro_recall",
        "pi_e","pi_n","pi_c",
        "n_e","n_n","n_c",
        "kappa_uniform","kappa_invfreq","rho_t_max","rho_t_p95",
        "gen_G_used","gen_pass","gen_pass_rate",
        "mean_H_student_unlab","mean_H_teacher_unlab",
        "mean_UCR_violation","mean_FR_violation",
        "LV_kl","rectify_delta_norm"
    ]
    log_path = os.path.join(run_output_dir, "round_log.csv")
    is_new_log = not os.path.exists(log_path)
    with open(log_path, "a", newline="", encoding="utf-8") as lf:
        writer = csv.writer(lf)
        if is_new_log:
            writer.writerow(ROUND_LOG_HEADER)

    # 读数据
    print(f"[*] Loading initial data...")
    L = read_jsonl(args.init_labeled)
    U = read_jsonl(args.unlabeled_pool)
    V = read_jsonl(args.val_path)
    T = read_jsonl(args.test_path) if args.test_path else []
    print(f"[*] Initial state: {len(L)} labeled, {len(U)} unlabeled, {len(V)} validation.")

    # 模型 & tokenizer
    print(f"[*] Loading model from: {args.model_dir}")
    clf = Classifier(args.model_dir, num_labels=3, fp16=True)
    tokenizer = clf.get_tokenizer()

    # LA/Focal 先验（基于 L）
    pri = count_by_label(L)
    prior_arr = np.array([pri["entailment"], pri["neutral"], pri["contradiction"]], dtype=np.float32)
    s = float(prior_arr.sum()) if prior_arr.sum() > 0 else 3.0
    prior_arr = prior_arr / s
    clf.set_imbalance_handles(
        class_priors={i: prior_arr[i] for i in range(3)},
        la_tau=args.la_tau,
        focal_gamma=args.focal_gamma
    )

    # Teacher / Rectifier / V-Head
    teacher = clone_teacher_from_student(clf)
    dim_z = clf.model.config.hidden_size
    rect = Rectifier(dim_z=dim_z, num_classes=3, hidden_mul=1.0, delta_scale=1.0).to(clf.device)
    vhead = VariationalHead(dim_z=dim_z).to(clf.device)
    opt_rect = torch.optim.AdamW(list(rect.parameters()) + list(vhead.parameters()), lr=2e-5, weight_decay=0.01)

    loss_cfg = VaGeRyLossCfg(epsilon_u=0.05, epsilon_f=0.03, lambda_u=1.0, delta_f=0.1, beta_v=1e-4)
    acfg = AdaptiveCfg(
        eps_u0=args.eps_u0, eps_f0=args.eps_f0,
        alpha_u=args.alpha_u, alpha_f=args.alpha_f,
        beta_u=args.beta_u, beta_f=args.beta_f,
        gamma_v=args.gamma_v, lambda_cls=args.lambda_cls,
    )

    # EMA 封装 rectified 模型
    rectified = RectifiedModel(student=clf, teacher=teacher, rectifier=rect)

    # 调度
    eta_list = [float(x) for x in args.eta_schedule.split(',')]
    while len(eta_list) < args.rounds:
        eta_list.append(eta_list[-1])

    minority_debt = {k: 0 for k in LABELS}

    # —— 主循环 ——
    for r in range(1, args.rounds + 1):
        print("\n" + "="*50)
        print(f"[*] Starting Active Learning Round {r}/{args.rounds} for {args.run_name}")
        print(f"[*] Current dataset size: {len(L)} labeled, {len(U)} unlabeled.")
        
        B = int(args.budget)
        eta = float(eta_list[r - 1])

        # Memorization
        print(f"[*] Round {r}: Memorization phase...")
        U_sub = U[:min(len(U), 3000)]
        memo_stats = memorization_phase(
            student=clf, teacher=teacher, rectifier=rect, vhead=vhead,
            unlabeled_rows=U_sub, batch_size=args.batch_size, tokenizer=tokenizer, max_len=args.max_len,
            optim_rect=opt_rect, loss_cfg=loss_cfg, max_steps=None, acfg=acfg,
            class_scarcity_getter=None, warmup_only_lv=(r <= args.warmup_rounds)
        )

        # Reception
        print(f"[*] Round {r}: Reception phase (training on {len(L)} samples)...")
        train_loader = build_loader(NliDataset(L, tokenizer=tokenizer, max_len=args.max_len),
                                    batch_size=args.batch_size, shuffle=True, tokenizer=tokenizer, max_len=args.max_len)
        val_loader = build_loader(NliDataset(V, tokenizer=tokenizer, max_len=args.max_len),
                                  batch_size=args.batch_size, shuffle=False, tokenizer=tokenizer, max_len=args.max_len)
        
        # 【修改】接收完整的验证集指标字典
        best_val_metrics = train_with_early_stop(
            clf, train_loader, val_loader,
            epochs=args.epochs, patience=args.patience, lr=args.lr
        )
        reception_phase(clf, teacher, rect, L, V, args.batch_size, tokenizer, args.max_len,
                        lr=args.lr, epochs=args.epochs, ema_m=args.ema_m)

        # 选样
        print(f"[*] Round {r}: Selection phase (budget B={B})...")
        sel_info = sampling.va_alvin_select(
            unlabeled_pool=U, labeled_rows=L, model=rectified,
            tokenizer=tokenizer, batch_size=args.batch_size, max_len=args.max_len,
            budget_b=B, eta=eta, floor_rho=args.floor_rho, cap_frac=args.cap_frac,
            mc_passes=args.mc_passes, boundary_tau=args.boundary_tau,
            residual_boost_frac=args.residual_boost_frac, minority_debt=minority_debt,
        )
        idx_selected = sel_info["selected_indices"]
        by_class = sel_info["by_class"]
        minority_debt = sel_info["minority_debt_next"]
        print(f"[*] Selected {len(idx_selected)} samples. Class distribution of new samples: {by_class}")

        
        selected_rows = [U[i] for i in idx_selected]
        
        # ... (伪标逻辑保持不变)

        # 【核心逻辑】 更新 L 和 U 数据集
        L.extend(selected_rows)
        # ... (伪标逻辑)
        for i in sorted(idx_selected, reverse=True):
            U.pop(i)
        
        print(f"[*] Round {r} finished. New dataset size: {len(L)} labeled, {len(U)} unlabeled.")

        # 【修改】指标计算和日志记录
        print("[*] Calculating comprehensive metrics and logging...")
        cnt_L = count_by_label(L)
        mins, _ = majority_minority_classes(cnt_L)
        minority_class = mins[0] if mins else "N/A"
        minority_idx = L2I.get(minority_class, -1)

        # 从 best_val_metrics 字典中提取详细指标
        minority_f1 = best_val_metrics.get("per_class_f1", [0,0,0])[minority_idx] if minority_idx != -1 else 0.0
        minority_recall = best_val_metrics.get("per_class_recall", [0,0,0])[minority_idx] if minority_idx != -1 else 0.0
        minority_precision = best_val_metrics.get("per_class_precision", [0,0,0])[minority_idx] if minority_idx != -1 else 0.0

        # 计算新选出样本的类别分布 (pi_e, pi_n, pi_c)
        pi_counts = count_by_label(selected_rows)
        denom = max(1, len(selected_rows))
        pi_e = pi_counts.get('entailment',0)/denom
        pi_n = pi_counts.get('neutral',0)/denom
        pi_c = pi_counts.get('contradiction',0)/denom
        
        cov_metrics = compute_coverage_metrics(selected_rows, U, clf, tokenizer, args.batch_size, args.max_len)

        # 准备写入日志的数据字典
        log_data = {
            "timestamp": timestamp_str(), "strategy": args.strategy, "mode": args.mode, "round": r,
            "labeled_total": len(L), "B": B,
            "minority_class": minority_class, "minority_f1": f"{minority_f1:.4f}",
            "minority_recall": f"{minority_recall:.4f}", "minority_precision": f"{minority_precision:.4f}",
            "macro_f1": f"{best_val_metrics.get('macro_f1', 0.0):.4f}", "accuracy": f"{best_val_metrics.get('accuracy', 0.0):.4f}",
            "macro_precision": f"{best_val_metrics.get('macro_precision', 0.0):.4f}", "macro_recall": f"{best_val_metrics.get('macro_recall', 0.0):.4f}",
            "pi_e": f"{pi_e:.4f}", "pi_n": f"{pi_n:.4f}", "pi_c": f"{pi_c:.4f}",
            "n_e": cnt_L.get("entailment", 0), "n_n": cnt_L.get("neutral", 0), "n_c": cnt_L.get("contradiction", 0),
            "kappa_uniform": f"{cov_metrics.get('kappa_uniform', 0.0):.6f}", "kappa_invfreq": f"{cov_metrics.get('kappa_invfreq', 0.0):.6f}",
            "rho_t_max": f"{cov_metrics.get('rho_t_max', 0.0):.6f}", "rho_t_p95": f"{cov_metrics.get('rho_t_p95', 0.0):.6f}",
            "gen_G_used": 0, "gen_pass": 0, "gen_pass_rate": f"{0.0:.4f}", # Generation logic not in this script
            "mean_H_student_unlab": f"{memo_stats.get('mean_H_student_unlab', 0.0):.6f}", "mean_H_teacher_unlab": f"{memo_stats.get('mean_H_teacher_unlab', 0.0):.6f}",
            "mean_UCR_violation": f"{memo_stats.get('mean_UCR_violation', 0.0):.6f}", "mean_FR_violation": f"{memo_stats.get('mean_FR_violation', 0.0):.6f}",
            "LV_kl": f"{memo_stats.get('LV_kl', 0.0):.6f}", "rectify_delta_norm": f"{memo_stats.get('rectify_delta_norm', 0.0):.6f}"
        }
        
        with open(log_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([log_data.get(h, "") for h in ROUND_LOG_HEADER])

        # 保存进度
        write_jsonl(os.path.join(run_output_dir, f"L_round_{r}.jsonl"), L)
        write_jsonl(os.path.join(run_output_dir, f"U_round_{r}.jsonl"), U)
        
        if not U:
            print("[*] Unlabeled pool exhausted. Halting.")
            break

    print("\n" + "="*50)
    print("[*] Active Learning Finished.")
    if T:
        print("[*] Evaluating on final test set...")
        test_loader = build_loader(NliDataset(T, tokenizer, args.max_len), args.batch_size, False, tokenizer, args.max_len)
        probs = rectified.predict_proba(test_loader)
        np.save(os.path.join(run_output_dir, "final_test_probs.npy"), probs)
        print(f"[*] Test probabilities saved to {run_output_dir}")

if __name__ == "__main__":
    main()
