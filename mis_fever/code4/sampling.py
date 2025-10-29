# -*- coding: utf-8 -*-
"""
统一的主动学习策略选择模块（增强版）：
- random: 随机采样
- entropy: 熵采样（全池）
- alvin: 类内插值 + KNN(欧氏距离) + 不确定性，从未标注池选样（基础版）
- short_simple: 短句生成（需要 llm_client），含 8/8 一致性校验与配平
- long_simple:  长前提 + 简单关系生成（需要 llm_client），含一致性校验
- difficulty / misclassified:（保留接口）
- va_alvin_select: 【增强】类感知方差重标化 + 探索/利用 + floor/cap +
                   双通道入选 + 边界回填 + 剩余方差二次配额 + 欠账跨轮补偿

依赖：
- model 需实现：
  * predict_proba(loader) -> np.ndarray [N, C]
  * encode(loader) -> np.ndarray [N, D]
  * get_tokenizer()
- 数据与工具：
  * NliDataset, build_loader
"""

from __future__ import annotations
from collections import defaultdict, Counter
from typing import List, Dict, Tuple, Optional

import numpy as np

from datasets import NliDataset
from train_utils import build_loader

LABELS = ["entailment", "neutral", "contradiction"]
LABEL_TO_ID = {l: i for i, l in enumerate(LABELS)}
ID_TO_LABEL = {i: l for l, i in LABEL_TO_ID.items()}

# -----------------------------
# 工具函数
# -----------------------------

def _entropy_from_probs(probs: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    probs = np.clip(probs, eps, 1.0)
    return -np.sum(probs * np.log(probs), axis=1)


def _build_loader(rows: List[Dict], tokenizer, max_len: int, batch_size: int, shuffle: bool = False):
    dataset = NliDataset(rows, tokenizer=tokenizer, max_len=max_len)
    return build_loader(dataset, batch_size=batch_size, shuffle=shuffle, tokenizer=tokenizer, max_len=max_len)


def _normalize_rows(rows: List[Dict]) -> List[Dict]:
    out = []
    for r in rows:
        out.append({
            "premise": r["premise"],
            "hypothesis": r["hypothesis"],
            **({"label": r["label"]} if "label" in r else {})
        })
    return out


def _class_counts(rows: List[Dict]) -> Dict[str, int]:
    cnt = {l: 0 for l in LABELS}
    for r in rows:
        if "label" in r and r["label"] in cnt:
            cnt[r["label"]] += 1
    return cnt


def _predict_class_ids(probs: np.ndarray) -> np.ndarray:
    return probs.argmax(axis=1)


def _top2_margin(probs: np.ndarray) -> np.ndarray:
    # top1 - top2 的差值（越小越靠近决策边界）
    part = np.partition(-probs, 1, axis=1)
    top1 = -part[:, 0]
    top2 = -part[:, 1]
    return top1 - top2


def _classwise_iqr_renorm(v: np.ndarray, y_pred: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """对每个预测类做 IQR(q25~q75) 归一化，再拼回全局。"""
    v = v.copy()
    for k in range(len(LABELS)):
        mask = (y_pred == k)
        if not np.any(mask):
            continue
        vk = v[mask]
        q25, q75 = np.percentile(vk, [25, 75])
        denom = max(q75 - q25, eps)
        v[mask] = (vk - q25) / denom
    # 再做一次全局 min-max 限制到 [0,1]
    v -= v.min()
    vmax = max(v.max(), eps)
    v /= vmax
    return v


def _compute_pool_stats(unlabeled: List[Dict], model, tokenizer, batch_size: int, max_len: int,
                        mc_passes: int = 3) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    返回：
      probs_mean: [N,C] rectified 概率的均值（用 MC-dropout 重复前向）
      var_score:  [N]   方差分数（类别概率方差和）
      y_pred:     [N]   预测类 id
      margin:     [N]   top1-top2 的概率差
    说明：这里默认 model.predict_proba() 已是 rectified 概率接口（或等价）。
    """
    rows = _normalize_rows(unlabeled)
    loader = _build_loader(rows, tokenizer, max_len, batch_size, shuffle=False)

    # MC-dropout：取均值 + 方差
    probs_runs = []
    for _ in range(max(1, mc_passes)):
        probs = model.predict_proba(loader)  # [N,C]
        probs_runs.append(probs)
    probs_stack = np.stack(probs_runs, axis=0)  # [T,N,C]
    probs_mean = probs_stack.mean(axis=0)
    probs_var = probs_stack.var(axis=0)
    var_score = probs_var.sum(axis=1)  # 类概率的总方差

    y_pred = _predict_class_ids(probs_mean)
    margin = _top2_margin(probs_mean)
    return probs_mean, var_score, y_pred, margin


def random_select(unlabeled_pool: List[Dict], budget_b: int, seed: int = 42) -> List[int]:
    rng = np.random.default_rng(seed)
    n = len(unlabeled_pool)
    if budget_b >= n:
        return list(range(n))
    return rng.choice(n, size=budget_b, replace=False).tolist()


def entropy_select(unlabeled_pool: List[Dict], model, tokenizer, batch_size: int, max_len: int, budget_b: int,
                   confidence_filter: Optional[float] = None) -> List[int]:
    rows = _normalize_rows(unlabeled_pool)
    loader = _build_loader(rows, tokenizer, max_len, batch_size, shuffle=False)
    probs = model.predict_proba(loader)
    entropy = _entropy_from_probs(probs)
    idxs = np.arange(len(unlabeled_pool))
    if confidence_filter is not None:
        max_p = probs.max(axis=1)
        mask = max_p <= confidence_filter
        idxs = idxs[mask]
        entropy = entropy[mask]
    order = np.argsort(-entropy)
    return idxs[order][:budget_b].tolist()

# -----------------------------
# VA-ALVIN（增强版）
# -----------------------------

def va_alvin_select(
    unlabeled_pool: List[Dict],
    labeled_rows: List[Dict],
    model,
    tokenizer,
    batch_size: int,
    max_len: int,
    budget_b: int,
    eta: float = 0.6,
    floor_rho: float = 0.5,
    cap_frac: float = 0.4,
    mc_passes: int = 3,
    boundary_tau: float = 0.1,
    residual_boost_frac: float = 0.15,
    minority_debt: Optional[Dict[str, int]] = None,
    rng: Optional[np.random.Generator] = None,
) -> Dict:
    """
    返回字典：{
      "selected_indices": [...],           # 最终选中的池索引
      "by_channel": {"exploit": [...], "explore": [...]},
      "by_class": {label: count, ...},
      "minority_debt_next": {label: int, ...}  # 跨轮欠账
    }
    """
    if rng is None:
        rng = np.random.default_rng(42)

    N = len(unlabeled_pool)
    # ✅ 修改：优雅处理空池或零预算，避免程序在日志写入前中断
    if N == 0 or budget_b <= 0:
        return {
            "selected_indices": [],
            "by_channel": {"exploit": [], "explore": []},
            "by_class": {k: 0 for k in LABELS},
            "minority_debt_next": (minority_debt or {k: 0 for k in LABELS}),
        }

    # 1) 统计类稀缺（基于已标注集）
    counts = _class_counts(labeled_rows)
    maxn = max(1, max(counts.values()))
    pi = {k: counts[k] / maxn for k in LABELS}  # π_k ∈ [0,1]

    # 2) 计算池上的 rectified 概率均值 + 方差分数 + 预测类 + 边界 margin
    probs, v_raw, y_pred, margin = _compute_pool_stats(
        unlabeled_pool, model, tokenizer, batch_size, max_len, mc_passes=mc_passes
    )

    # 类感知 IQR 归一化的方差分数
    v_hat = _classwise_iqr_renorm(v_raw, y_pred)

    # 3) 利用-探索划分
    B_exp = int(round(eta * budget_b))
    B_explore = budget_b - B_exp

    # 4) 类内 floor/cap（利用部分）
    inv_pi_sum = sum(1.0 - pi[k] for k in LABELS)
    minority_debt = minority_debt or {k: 0 for k in LABELS}

    floor = {}
    for k in LABELS:
        base = (1.0 - pi[k]) / max(inv_pi_sum, 1e-8)
        need = int(np.floor(floor_rho * B_exp * base))
        need = max(1, need)  # 至少 1
        # 欠账补偿（上一轮未达成的 floor）
        need += int(minority_debt.get(k, 0))
        floor[k] = need

    cap = {k: int(np.ceil(cap_frac * B_exp)) for k in LABELS}

    # 5) 双通道入选：
    #   (a) 各类内根据 v_hat 排序，先取 floor[k]，得到第一通道候选
    #   (b) 余量进入全局第二通道，仍按 v_hat 排序并受 cap 约束
    idx_all = np.arange(N)
    chosen = []
    used = np.zeros(N, dtype=bool)

    # (a) 类内 Top-K（floor）
    class_bins = {k: np.where(y_pred == LABELS.index(k))[0] for k in LABELS}
    per_class_taken = {k: 0 for k in LABELS}

    for k in LABELS:
        idxs = class_bins[k]
        if idxs.size == 0:
            continue
        order = np.argsort(-v_hat[idxs])
        take = min(floor[k], idxs.size)
        sel = idxs[order[:take]]
        chosen.extend(sel.tolist())
        used[sel] = True
        per_class_taken[k] += take

    # (b) 全局 Top-M（二通道，cap 约束）
    remain = np.where(~used)[0]
    order_global = np.argsort(-v_hat[remain])
    for i in order_global:
        if len(chosen) >= B_exp:
            break
        j = remain[i]
        k = LABELS[y_pred[j]]
        if per_class_taken[k] >= cap[k]:
            continue
        chosen.append(int(j))
        used[j] = True
        per_class_taken[k] += 1

    # 6) 类级“剩余方差”二次配额（对利用部分的 10–20% 做微调）
    extra_quota = int(round(residual_boost_frac * B_exp))
    if extra_quota > 0:
        # 计算每类剩余候选的平均方差（在未使用集合中）
        v_bar = {}
        for k in LABELS:
            rest = np.intersect1d(np.where(~used)[0], class_bins[k], assume_unique=False)
            v_bar[k] = float(v_hat[rest].mean()) if rest.size > 0 else 0.0
        s = sum(v_bar.values())
        if s > 0:
            # 追加名额按 v_bar 比例分配
            for k in sorted(LABELS, key=lambda x: -v_bar[x]):
                if extra_quota <= 0:
                    break
                rest = np.intersect1d(np.where(~used)[0], class_bins[k], assume_unique=False)
                if rest.size == 0:
                    continue
                # 取若干（不打破 cap）
                add_cap = max(0, cap[k] - per_class_taken[k])
                if add_cap <= 0:
                    continue
                # 从该类剩余中按 v_hat 排序取 1~add_cap
                order_k = np.argsort(-v_hat[rest])
                take = min(add_cap, extra_quota, rest.size)
                sel = rest[order_k[:take]]
                chosen.extend(sel.tolist())
                used[sel] = True
                per_class_taken[k] += take
                extra_quota -= take

    # 7) 边界相近回填（若利用通道尚未达到 B_exp）
    if len(chosen) < B_exp:
        remain = np.where(~used)[0]
        # margin 小 → 靠近边界；我们优先考虑当前最稀缺类
        scarce_order = sorted(LABELS, key=lambda x: pi[x])  # pi 越小越稀缺
        for k in scarce_order:
            if len(chosen) >= B_exp:
                break
            rest_k = np.intersect1d(remain, class_bins[k], assume_unique=False)
            if rest_k.size == 0:
                continue
            mask = (margin[rest_k] <= boundary_tau)
            cand = rest_k[mask]
            if cand.size == 0:
                continue
            # 在靠近边界者中，仍按 v_hat 排序以保留不确定性价值
            order = np.argsort(-v_hat[cand])
            for j in cand[order]:
                if len(chosen) >= B_exp:
                    break
                # 不打破 cap
                if per_class_taken[k] >= cap[k]:
                    break
                chosen.append(int(j))
                used[j] = True
                per_class_taken[k] += 1

    # 8) 探索通道（B_explore）：低密度/未覆盖区域随机（或用几何覆盖近似）
    # 这里给出简洁实现：按 margin 升序的一部分 + 随机混入，近似“不同于已选”的区域
    remain = np.where(~used)[0]
    explore = []
    if B_explore > 0 and remain.size > 0:
        # 取 50% 的最小 margin（边界附近的多样性），50% 随机
        half = B_explore // 2
        order_margin = np.argsort(margin[remain])  # 小→近边界
        near_bd = remain[order_margin[:min(half, remain.size)]]
        used[near_bd] = True
        explore.extend(near_bd.tolist())
        remain = np.where(~used)[0]
        if len(explore) < B_explore and remain.size > 0:
            k = B_explore - len(explore)
            pick = rng.choice(remain, size=min(k, remain.size), replace=False)
            used[pick] = True
            explore.extend(pick.tolist())

    selected = chosen + explore
    selected = selected[:budget_b]

    # 欠账更新：若某类 floor 未满足，记录差额
    minority_debt_next = {}
    for k in LABELS:
        achieved = per_class_taken[k]
        target = floor[k]
        minority_debt_next[k] = max(0, target - achieved)

    # 统计
    by_class = Counter([LABELS[y_pred[i]] for i in selected])

    return {
        "selected_indices": selected,
        "by_channel": {"exploit": chosen[:B_exp], "explore": explore},
        "by_class": dict(by_class),
        "minority_debt_next": minority_debt_next,
    }

