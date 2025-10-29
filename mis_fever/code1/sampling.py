# -*- coding: utf-8 -*-
"""
统一的主动学习策略选择模块：
- random: 随机采样
- entropy: 熵采样（全池）
- alvin: 类内插值 + KNN(欧氏距离) + 不确定性，从未标注池选样
- short_simple: 短句生成（需要 llm_client），含 8/8 一致性校验与配平
- long_simple: 长前提 + 简单关系生成（需要 llm_client），含一致性校验
- difficulty: 难度分数采样（需要有标签的候选集合 + llm_client）
- misclassified: 误判样本采样（需要有标签的候选集合 + model）

依赖：
- 你自己的模型需实现：
  * model.predict_proba(loader) -> np.ndarray [N, C]
  * model.encode(loader) -> np.ndarray [N, D] (CLS/句向量或其他句向量)

- 数据集/工具：
  * NliDataset, build_loader
"""

from __future__ import annotations
from collections import defaultdict, Counter
from typing import List, Dict, Tuple, Optional, Callable

import numpy as np

from datasets import NliDataset
from train_utils import build_loader


# --------------------------------
# 常量 & Label 工具
# --------------------------------

LABELS = ["entailment", "neutral", "contradiction"]
LABEL_TO_ID = {l: i for i, l in enumerate(LABELS)}
ID_TO_LABEL = {i: l for l, i in LABEL_TO_ID.items()}


def _require_label_fields(rows: List[Dict]):
    for r in rows:
        assert "label" in r, "该策略需要 rows 内含有 'label' 字段"

def _label_to_str(v):
    # 允许传入 0/1/2 或 "entailment"/"neutral"/"contradiction"
    if isinstance(v, int):
        return ID_TO_LABEL[v]
    return str(v)


# --------------------------------
# 工具函数
# --------------------------------

def _entropy_from_probs(probs: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """H(p) = -sum_i p_i log(p_i)."""
    probs = np.clip(probs, eps, 1.0)
    return -np.sum(probs * np.log(probs), axis=1)


def _build_loader_from_rows(rows: List[Dict], tokenizer, max_len: int, batch_size: int, shuffle: bool = False):
    dataset = NliDataset(rows, tokenizer=tokenizer, max_len=max_len)
    return build_loader(dataset, batch_size=batch_size, shuffle=shuffle, tokenizer=tokenizer, max_len=max_len)


def _normalize_rows_for_subset(rows: List[Dict]) -> List[Dict]:
    """确保子集行包含 'premise'/'hypothesis'/'label'(若有) 字段。"""
    norm = []
    for r in rows:
        norm.append({
            "premise": r["premise"],
            "hypothesis": r["hypothesis"],
            **({"label": r["label"]} if "label" in r else {})
        })
    return norm


def _balance_take_by_label(rows: List[Dict], per_class_target: Dict[str, int]) -> List[Dict]:
    """从 rows 中按标签配平抽取，直到各类达到 per_class_target 或 rows 用尽。"""
    buckets = defaultdict(list)
    for r in rows:
        l = r.get("label")
        if l in LABEL_TO_ID:
            buckets[l].append(r)
    out = []
    for lab, need in per_class_target.items():
        got = buckets[lab][:need]
        out.extend(got)
    return out


# --------------------------------
# 基础策略：Random & Entropy
# --------------------------------

def random_select(unlabeled_pool: List[Dict], budget_b: int, seed: int = 42) -> List[int]:
    rng = np.random.default_rng(seed)
    n = len(unlabeled_pool)
    if budget_b >= n:
        return list(range(n))
    return rng.choice(n, size=budget_b, replace=False).tolist()


def entropy_select(unlabeled_pool: List[Dict], model, tokenizer, batch_size: int, max_len: int, budget_b: int,
                   confidence_filter: Optional[float] = None) -> List[int]:
    """
    计算熵 H(p) = -sum_i p_i log p_i，按熵降序取 top-b。
    可选：confidence_filter，如果 max(p) > 阈值，则认为过于确定（可剔除）。
    """
    dataset = NliDataset(unlabeled_pool, tokenizer=tokenizer, max_len=max_len)
    loader = build_loader(dataset, batch_size=batch_size, shuffle=False, tokenizer=tokenizer, max_len=max_len)
    probs = model.predict_proba(loader)  # [N, C]
    entropy = _entropy_from_probs(probs)

    idxs = np.arange(len(unlabeled_pool))
    if confidence_filter is not None:
        max_p = probs.max(axis=1)
        mask = max_p <= confidence_filter
        idxs = idxs[mask]
        entropy = entropy[mask]

    order = np.argsort(-entropy)  # descending
    selected = idxs[order][:budget_b].tolist()
    return selected


# --------------------------------
# 生成类策略：通用一致性校验 & 生成框架
# --------------------------------

def _judge_label_many(llm_client, sample: Dict, n_votes: int = 8) -> Optional[str]:
    """
    对一个样本做 n_votes 次判别，若全一致则返回该标签，否则返回 None。
    期望 llm_client.judge_label(sample) -> str in {"entailment","neutral","contradiction"}。
    """
    votes = []
    for _ in range(n_votes):
        if hasattr(llm_client, "judge_label"):
            y = llm_client.judge_label(sample)
        else:
            # 回退：若没有 judge_label，尝试 generic 调用
            y = llm_client.judge(sample)  # 需由用户在 llm_client 中适配
        y = str(y).lower()
        if y not in LABEL_TO_ID:
            return None
        votes.append(y)
    ok = len(set(votes)) == 1
    return votes[0] if ok else None


def _generate_loop_with_consensus(
    llm_client,
    gen_callable: Callable[[], Dict],
    n_items: int,
    n_votes: int = 8,
    max_trials: int = 2000,
    target_per_class: Optional[Dict[str, int]] = None,
) -> List[Dict]:
    """
    通用生成主循环：
      - 调用 gen_callable() 产生候选样本（需包含 premise/hypothesis）
      - 一致性判别 n_votes（8/8）
      - 可选类别配平 target_per_class
    """
    accepted: List[Dict] = []
    trials = 0

    def need_more() -> bool:
        if target_per_class is None:
            return len(accepted) < n_items
        cnt = Counter([s["label"] for s in accepted])
        for lab, tgt in target_per_class.items():
            if cnt[lab] < tgt:
                return True
        return False

    while need_more() and trials < max_trials:
        trials += 1
        try:
            cand = gen_callable()
            assert "premise" in cand and "hypothesis" in cand
        except Exception:
            continue

        y = _judge_label_many(llm_client, cand, n_votes=n_votes)
        if y is None:
            continue
        cand["label"] = y

        if target_per_class is not None:
            cnt = Counter([s["label"] for s in accepted])
            if cnt[y] >= target_per_class.get(y, 0):
                continue

        accepted.append(cand)

    if target_per_class is None and len(accepted) > n_items:
        accepted = accepted[:n_items]
    return accepted


# ----- Short & Simple -----

def short_simple_generate(
    llm_client,
    n_items: int,
    n_votes: int = 8,
    balance: bool = True,
    max_trials: int = 2000
) -> List[Dict]:
    """
    论文实现：短前提+短假设，LLM自标；做 n_votes=8 的一致性校验；可选按类配平。
    期望 llm_client 实现：
      - generate_short_simple_candidate() -> {"premise","hypothesis"}
      - 或 generate_short_simple(n_items=?, ...) 批量；此处做了 call 适配。
    """
    target = None
    if balance:
        base = n_items // 3
        extra = n_items - base * 3
        target = {lab: base for lab in LABELS}
        for i in range(extra):
            target[LABELS[i]] += 1

    def gen_one():
        if hasattr(llm_client, "generate_short_simple_candidate"):
            return llm_client.generate_short_simple_candidate()
        elif hasattr(llm_client, "generate_short_simple"):
            out = llm_client.generate_short_simple(n_items=1, max_trials=10)
            return out[0] if out else {}
        else:
            return llm_client.generate(template="short_simple")

    return _generate_loop_with_consensus(
        llm_client, gen_one, n_items=n_items, n_votes=n_votes, max_trials=max_trials, target_per_class=target
    )


# ----- Long & Simple -----

def long_simple_generate(
    llm_client,
    n_items: int,
    n_votes: int = 8,
    balance: bool = True,
    max_trials: int = 4000
) -> List[Dict]:
    """
    论文实现：4句前提 + 简单关系假设 + 一致性校验。
    期望 llm_client 提供：
      - generate_long_simple_candidate()
      - 或 generate_long_simple(n_items=?)
      - 否则使用通用模板 "long_simple"
    """
    target = None
    if balance:
        base = n_items // 3
        extra = n_items - base * 3
        target = {lab: base for lab in LABELS}
        for i in range(extra):
            target[LABELS[i]] += 1

    def gen_one():
        if hasattr(llm_client, "generate_long_simple_candidate"):
            return llm_client.generate_long_simple_candidate()
        elif hasattr(llm_client, "generate_long_simple"):
            out = llm_client.generate_long_simple(n_items=1, max_trials=10)
            return out[0] if out else {}
        else:
            return llm_client.generate(template="long_simple")  # 需在客户端侧保证“长前提+简单关系”

    return _generate_loop_with_consensus(
        llm_client, gen_one, n_items=n_items, n_votes=n_votes, max_trials=max_trials, target_per_class=target
    )


# --------------------------------
# 采样类策略：ALVIN（保留你的实现）
# --------------------------------

def _flag_minority_majority_by_dynamics(
    labeled_rows: List[Dict],
    dynamics: Dict[int, List[bool]],
    min_correct_frac: float = 0.4,
) -> Tuple[Dict[int, bool], Dict[int, str]]:
    """
    基于训练动态判定少数派（True）/多数派（False）。
    dynamics: {local_index_in_labeled_rows: [bool, bool, ...]}  每轮是否预测正确
    返回:
      is_minority: {idx -> bool}
      class_of_idx: {idx -> label_str}
    """
    is_minority = {}
    class_of_idx = {}
    for i, row in enumerate(labeled_rows):
        y = row["label"]
        class_of_idx[i] = y
        hist = dynamics.get(i, [])
        if len(hist) == 0:
            is_minority[i] = False
        else:
            frac = sum(hist) / float(len(hist))
            is_minority[i] = (frac < min_correct_frac)
    return is_minority, class_of_idx


def _flag_minority_majority_by_current_model(
    labeled_rows: List[Dict],
    model,
    tokenizer,
    batch_size: int,
    max_len: int,
    low_conf_thresh: float = 0.5,
) -> Tuple[Dict[int, bool], Dict[int, str]]:
    """
    若无 dynamics，则用当前模型在有标注集上的预测近似：
      - 预测错误样本视作“强少数派”；
      - 预测正确但 max_p < 低置信度阈值的，也视作“弱少数派”；
    其他归为多数派。
    """
    rows = _normalize_rows_for_subset(labeled_rows)
    loader = _build_loader_from_rows(rows, tokenizer, max_len, batch_size, shuffle=False)
    probs = model.predict_proba(loader)  # [M, C]
    pred = probs.argmax(axis=1)
    max_p = probs.max(axis=1)

    is_minority = {}
    class_of_idx = {}
    for i, r in enumerate(labeled_rows):
        y = _label_to_str(r["label"])  # string
        class_of_idx[i] = y
        minority = (pred[i] != LABEL_TO_ID[y]) or (max_p[i] < low_conf_thresh)
        is_minority[i] = minority
    return is_minority, class_of_idx


def _classwise_min_maj_indices(
    labeled_rows: List[Dict],
    is_minority: Dict[int, bool],
    class_of_idx: Dict[int, str],
) -> Dict[str, Dict[str, List[int]]]:
    """
    返回 per-class 字典（键为标签字符串）：
      class_to_indices[label] = {"min": [...], "maj": [...]}
    """
    out: Dict[str, Dict[str, List[int]]] = defaultdict(lambda: {"min": [], "maj": []})
    for i in range(len(labeled_rows)):
        c = _label_to_str(class_of_idx[i])  # label string
        if is_minority[i]:
            out[c]["min"].append(i)
        else:
            out[c]["maj"].append(i)
    return out


def _encode_rows(rows: List[Dict], model, tokenizer, batch_size: int, max_len: int) -> np.ndarray:
    rows = _normalize_rows_for_subset(rows)
    loader = _build_loader_from_rows(rows, tokenizer, max_len, batch_size, shuffle=False)
    return model.encode(loader)  # [N, D]  (np.ndarray)


def _l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True) + eps
    return x / n


def _euclidean_knn_indices_numpy(
    anchors: np.ndarray,      # [A, D]
    pool_enc: np.ndarray,     # [N, D]
    top_k: int,
    batch: int = 2048,
    normalize: bool = False,
) -> List[int]:
    """
    纯 numpy 欧氏距离 KNN。可选先 L2 归一化（normalize=True）。
    返回去重后的候选索引列表。
    """
    if anchors.size == 0 or pool_enc.size == 0:
        return []
    A = _l2_normalize(anchors) if normalize else anchors
    P = _l2_normalize(pool_enc) if normalize else pool_enc

    P_sq = np.sum(P * P, axis=1)  # [N]
    cand = set()
    N = P.shape[0]
    for i in range(A.shape[0]):
        a = A[i:i+1]               # [1, D]
        a_sq = np.sum(a * a, axis=1)  # [1]
        d2 = np.empty((N,), dtype=np.float32)
        ptr = 0
        while ptr < N:
            end = min(ptr + batch, N)
            Pa = P[ptr:end]
            dot = Pa @ a.T          # [chunk, 1]
            d2[ptr:end] = P_sq[ptr:end] - 2.0 * dot.ravel() + a_sq[0]
            ptr = end
        if top_k >= N:
            top_idx = np.argsort(d2)
        else:
            part = np.argpartition(d2, kth=min(top_k, N-1))[:top_k]
            top_idx = part[np.argsort(d2[part])]
        for t in top_idx:
            cand.add(int(t))
    return list(cand)


def _euclidean_knn_indices_faiss(
    anchors: np.ndarray,
    pool_enc: np.ndarray,
    top_k: int,
    normalize: bool = False,
    use_gpu: bool = False,
) -> Optional[List[int]]:
    """
    使用 FAISS 的欧氏距离 KNN。若导入失败或环境不支持，则返回 None。
    """
    try:
        import faiss  # type: ignore
    except Exception:
        return None

    A = _l2_normalize(anchors) if normalize else anchors
    P = _l2_normalize(pool_enc) if normalize else pool_enc

    d = P.shape[1]
    index = faiss.IndexFlatL2(d)
    if use_gpu:
        try:
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, 0, index)
        except Exception:
            pass

    P32 = np.ascontiguousarray(P.astype(np.float32))
    index.add(P32)

    A32 = np.ascontiguousarray(A.astype(np.float32))
    _, idxs = index.search(A32, min(top_k, P.shape[0]))  # [A, top_k]
    return list(sorted(set(map(int, idxs.ravel()))))


def alvin_select(
    unlabeled_pool: List[Dict],
    labeled_set: List[Dict],
    model,
    tokenizer,
    batch_size: int,
    max_len: int,
    budget_b: int,
    dynamics: Optional[Dict[int, List[bool]]] = None,  # 若提供，优先用训练动态划分少数/多数
    low_conf_thresh: float = 0.5,                      # 无动态时的低置信阈值
    anchors_per_class: int = 20,                       # 每类锚点数（低资源可降到 10）
    alpha: float = 2.0,                                # ★ 论文推荐：Beta(alpha, alpha)
    knn_k: int = 15,                                   # ★ 论文推荐：每个锚点取 K=15 个近邻
    rng_seed: int = 42,
    normalize_encodings: bool = False,                 # 论文用欧氏距离；可选归一化
    use_faiss: bool = False,                           # 若环境支持可设 True
    faiss_use_gpu: bool = False,                       # 有 GPU 的 faiss 才能用 True
    fallback_confidence_filter: Optional[float] = None,
) -> List[int]:
    """
    ALVIN 主流程：
      1) 少数/多数划分（优先用 dynamics；否则基于当前模型表现近似）
      2) 类内插值产锚点（lambda~Beta(alpha,alpha)）
      3) 锚点 KNN（欧氏距离）召回未标注候选
      4) 候选上做熵排序，取 top-b
      5) 候选不足时回退 entropy_select
    """
    rng = np.random.default_rng(rng_seed)

    if len(unlabeled_pool) == 0:
        return []

    # 1) 划分少数/多数
    if dynamics is not None:
        is_min, class_of_idx = _flag_minority_majority_by_dynamics(
            labeled_rows=labeled_set, dynamics=dynamics, min_correct_frac=0.4
        )
    else:
        is_min, class_of_idx = _flag_minority_majority_by_current_model(
            labeled_rows=labeled_set, model=model, tokenizer=tokenizer,
            batch_size=batch_size, max_len=max_len, low_conf_thresh=low_conf_thresh
        )

    class_to_groups = _classwise_min_maj_indices(labeled_set, is_min, class_of_idx)
    any_valid = any(len(v["min"]) > 0 and len(v["maj"]) > 0 for v in class_to_groups.values())
    if not any_valid:
        return entropy_select(
            unlabeled_pool, model, tokenizer, batch_size, max_len, budget_b,
            confidence_filter=fallback_confidence_filter
        )

    # 2) 编码向量
    labeled_enc = _encode_rows(labeled_set, model, tokenizer, batch_size, max_len)   # [M, D]
    pool_enc = _encode_rows(unlabeled_pool, model, tokenizer, batch_size, max_len)   # [N, D]

    # 3) 按类创建锚点
    anchors = []
    for _, groups in class_to_groups.items():
        min_idx = groups["min"]
        maj_idx = groups["maj"]
        if len(min_idx) == 0 or len(maj_idx) == 0:
            continue
        enc_min = labeled_enc[np.array(min_idx)]
        enc_maj = labeled_enc[np.array(maj_idx)]
        for _ in range(anchors_per_class):
            i = rng.integers(0, len(enc_min))
            j = rng.integers(0, len(enc_maj))
            lam = rng.beta(alpha, alpha)
            a = lam * enc_min[i] + (1.0 - lam) * enc_maj[j]
            anchors.append(a)

    if len(anchors) == 0:
        return entropy_select(
            unlabeled_pool, model, tokenizer, batch_size, max_len, budget_b,
            confidence_filter=fallback_confidence_filter
        )
    anchors = np.stack(anchors, axis=0)  # [A, D]

    # 4) KNN（欧氏距离）召回候选
    cand_indices: Optional[List[int]] = None
    if use_faiss:
        cand_indices = _euclidean_knn_indices_faiss(
            anchors=anchors, pool_enc=pool_enc, top_k=knn_k,
            normalize=normalize_encodings, use_gpu=faiss_use_gpu
        )
    if cand_indices is None:
        cand_indices = _euclidean_knn_indices_numpy(
            anchors=anchors, pool_enc=pool_enc, top_k=knn_k,
            batch=2048, normalize=normalize_encodings
        )

    if len(cand_indices) == 0:
        return entropy_select(
            unlabeled_pool, model, tokenizer, batch_size, max_len, budget_b,
            confidence_filter=fallback_confidence_filter
        )

    # 5) 候选做熵排序
    cand_rows = [unlabeled_pool[i] for i in cand_indices]
    cand_loader = _build_loader_from_rows(cand_rows, tokenizer, max_len, batch_size, shuffle=False)
    cand_probs = model.predict_proba(cand_loader)  # [Nc, C]
    cand_entropy = _entropy_from_probs(cand_probs)
    order = np.argsort(-cand_entropy)  # 降序
    chosen = [cand_indices[i] for i in order[:min(budget_b, len(cand_indices))]]

    if len(chosen) < budget_b:
        remaining = sorted(set(range(len(unlabeled_pool))) - set(cand_indices))
        if remaining:
            rem_rows = [unlabeled_pool[i] for i in remaining]
            rem_loader = _build_loader_from_rows(rem_rows, tokenizer, max_len, batch_size, shuffle=False)
            rem_probs = model.predict_proba(rem_loader)
            rem_entropy = _entropy_from_probs(rem_probs)
            rem_order = np.argsort(-rem_entropy)
            need = budget_b - len(chosen)
            chosen += [remaining[i] for i in rem_order[:min(need, len(remaining))]]

    return chosen


# --------------------------------
# 采样类策略：Difficulty Score & Misclassified
# --------------------------------

def difficulty_score_select(
    candidate_labeled_set: List[Dict],
    llm_client,
    per_class_k: int = 10,
    fewshot_examples: Optional[List[Dict]] = None,
) -> List[int]:
    """
    难度分数采样（论文）：对带标签候选集逐条打难度 1..10，按“每类 top-K 难例”返回索引。
    期望 llm_client 提供：
      - score_difficulty(sample, fewshot_examples=None) -> int(1..10)
      - 或 score_difficulty_batch(samples, fewshot_examples=None) -> List[int]
    """
    _require_label_fields(candidate_labeled_set)
    n = len(candidate_labeled_set)
    scores = np.zeros(n, dtype=np.float32)

    if hasattr(llm_client, "score_difficulty_batch"):
        scores = np.asarray(llm_client.score_difficulty_batch(candidate_labeled_set, fewshot_examples=fewshot_examples))
        assert len(scores) == n
    else:
        vals = []
        for s in candidate_labeled_set:
            sc = llm_client.score_difficulty(s, fewshot_examples=fewshot_examples)
            vals.append(float(sc))
        scores = np.asarray(vals, dtype=np.float32)

    by_lab = defaultdict(list)
    for i, r in enumerate(candidate_labeled_set):
        lab = _label_to_str(r["label"])
        by_lab[r["label"]].append(i)

    selected = []
    for lab, idxs in by_lab.items():
        order = sorted(idxs, key=lambda j: -scores[j])
        selected.extend(order[:per_class_k])
    return selected


def misclassified_select(
    candidate_labeled_set: List[Dict],
    model,
    tokenizer,
    batch_size: int,
    max_len: int,
    per_class_k: int = 10,
) -> List[int]:
    """
    误判样本采样：用当前模型在候选集上推理，选择“预测 != 真值”的样本，
    按真实标签分组，每类最多取 K 条。
    注意：该策略在论文中被发现包含较高噪声（需谨慎）。
    """
    _require_label_fields(candidate_labeled_set)
    rows = _normalize_rows_for_subset(candidate_labeled_set)
    loader = _build_loader_from_rows(rows, tokenizer, max_len, batch_size, shuffle=False)
    probs = model.predict_proba(loader)  # [N, C]
    pred = probs.argmax(axis=1)

    by_lab_mis = defaultdict(list)
    for i, r in enumerate(candidate_labeled_set):
        y_str = _label_to_str(r["label"])
        y = LABEL_TO_ID[y_str]
        if pred[i] != y:
            by_lab_mis[y_str].append(i)

    selected = []
    for lab, idxs in by_lab_mis.items():
        selected.extend(idxs[:per_class_k])
    return selected


# --------------------------------
# 统一入口
# --------------------------------

def select(
    strategy: str,
    unlabeled_pool: List[Dict],
    model,               # Classifier
    tokenizer,           # HF tokenizer
    batch_size: int,
    max_len: int,
    budget_b: int,
    seed: int = 42,
    llm_client=None,
    confidence_filter: Optional[float] = None,
    llm_max_trials: int = 2000,
    # ---- ALVIN 额外参数 ----
    labeled_set: Optional[List[Dict]] = None,
    dynamics: Optional[Dict[int, List[bool]]] = None,
    low_conf_thresh: float = 0.5,
    anchors_per_class: int = 20,
    alpha: float = 2.0,             # ★ 论文默认
    knn_k: int = 15,                # ★ 论文默认
    rng_seed: int = 42,
    normalize_encodings: bool = False,
    use_faiss: bool = False,
    faiss_use_gpu: bool = False,
    # ---- 论文新增策略所需 ----
    candidate_labeled_set: Optional[List[Dict]] = None,  # 用于 difficulty / misclassified
    per_class_k: int = 10,
    # ---- 生成一致性参数 ----
    n_votes: int = 8,
    balance_generated: bool = True,
) -> Tuple[Optional[List[int]], Optional[List[Dict]]]:
    """
    统一入口：
    - 返回 (indices, None) 表示从 pool 或 candidate 上“选中的索引”（注意：difficulty/misclassified 对应 candidate_labeled_set）
    - 返回 (None, new_samples) 表示策略直接“生成/判定”的新样本（不属于 pool）
    """
    s = strategy.lower()

    # ------ 传统池采样 ------
    if s == "random":
        idxs = random_select(unlabeled_pool, budget_b, seed=seed)
        return idxs, None

    if s == "entropy":
        idxs = entropy_select(
            unlabeled_pool, model, tokenizer, batch_size, max_len, budget_b,
            confidence_filter=confidence_filter
        )
        return idxs, None

    if s == "alvin":
        assert labeled_set is not None and len(labeled_set) > 0, "ALVIN 需要 labeled_set（带标签的已标注样本）"
        idxs = alvin_select(
            unlabeled_pool=unlabeled_pool,
            labeled_set=labeled_set,
            model=model,
            tokenizer=tokenizer,
            batch_size=batch_size,
            max_len=max_len,
            budget_b=budget_b,
            dynamics=dynamics,
            low_conf_thresh=low_conf_thresh,
            anchors_per_class=anchors_per_class,
            alpha=alpha,
            knn_k=knn_k,
            rng_seed=rng_seed,
            normalize_encodings=normalize_encodings,
            use_faiss=use_faiss,
            faiss_use_gpu=faiss_use_gpu,
            fallback_confidence_filter=confidence_filter,
        )
        return idxs, None

    # ------ 论文新增的候选集采样（需要 candidate_labeled_set） ------
    if s in ("difficulty", "difficulty_score", "difficulty_score_sampling"):
        assert candidate_labeled_set is not None and len(candidate_labeled_set) > 0, \
            "difficulty 需要 candidate_labeled_set（带标签候选集）"
        assert llm_client is not None, "difficulty 需要 llm_client"
        idxs = difficulty_score_select(
            candidate_labeled_set=candidate_labeled_set,
            llm_client=llm_client,
            per_class_k=per_class_k,
            fewshot_examples=None
        )
        return idxs, None

    if s in ("misclassified", "misclassified_sampling"):
        assert candidate_labeled_set is not None and len(candidate_labeled_set) > 0, \
            "misclassified 需要 candidate_labeled_set（带标签候选集）"
        idxs = misclassified_select(
            candidate_labeled_set=candidate_labeled_set,
            model=model,
            tokenizer=tokenizer,
            batch_size=batch_size,
            max_len=max_len,
            per_class_k=per_class_k
        )
        return idxs, None

    # ------ 生成策略（返回“新样本”列表） ------
    if s in ("short_simple", "short&simple", "short-simple", "shortsimple"):
        assert llm_client is not None, "Short&Simple 需要 llm_client"
        new_samples = short_simple_generate(
            llm_client, n_items=budget_b, n_votes=n_votes,
            balance=balance_generated, max_trials=llm_max_trials
        )
        return None, new_samples

    if s in ("long_simple", "long-simple"):
        assert llm_client is not None, "Long&Simple 需要 llm_client"
        new_samples = long_simple_generate(
            llm_client, n_items=budget_b, n_votes=n_votes,
            balance=balance_generated, max_trials=max(llm_max_trials, 4000)
        )
        return None, new_samples

    raise ValueError(f"Unknown strategy: {strategy}")
