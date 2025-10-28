import math
import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import defaultdict

from datasets import NliDataset, LABEL2ID
from train_utils import build_loader

# 注意：本文件只实现策略逻辑；需要从外部传入 model/tokenizer/llm_client 等。


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
    loader = build_loader(dataset, batch_size=batch_size, shuffle=False, tokenizer=tokenizer)
    probs = model.predict_proba(loader)  # [N, C]
    eps = 1e-12
    entropy = -np.sum(probs * np.log(probs + eps), axis=1)  # [N]

    idxs = np.arange(len(unlabeled_pool))
    if confidence_filter is not None:
        max_p = probs.max(axis=1)
        mask = max_p <= confidence_filter
        idxs = idxs[mask]
        entropy = entropy[mask]

    order = np.argsort(-entropy)  # descending
    selected = idxs[order][:budget_b].tolist()
    return selected


def group_by_premise(unlabeled_pool: List[Dict]) -> Dict[str, List[int]]:
    """
    按 premise 分簇，返回 {premise_text: [indices...]}
    """
    buckets = defaultdict(list)
    for i, row in enumerate(unlabeled_pool):
        prem = row["premise"]
        buckets[prem].append(i)
    return buckets


def hyp_concat_generate(llm_client, unlabeled_pool: List[Dict], n_items: int) -> List[Dict]:
    """
    从 pool 中选择有 >=2 条 hypothesis 的同一 premise，随机取两条拼接后交给 LLM 判标签。
    生成的新样本直接带标签返回（source="concat"），不从 pool 扣减（因为生成的是新组合）。
    """
    import random
    buckets = group_by_premise(unlabeled_pool)
    candidates = [prem for prem, idxs in buckets.items() if len(idxs) >= 2]
    if not candidates:
        return []

    results = []
    random.shuffle(candidates)
    # 遍历 premise，尽量覆盖不同簇
    for prem in candidates:
        if len(results) >= n_items:
            break
        idxs = buckets[prem]
        if len(idxs) < 2:
            continue
        # 随机取两条 hypothesis
        a, b = random.sample(idxs, 2)
        hyp_a = unlabeled_pool[a]["hypothesis"]
        hyp_b = unlabeled_pool[b]["hypothesis"]
        # 拼接策略：这里采用简单并列（你可按论文改为更复杂模板）
        hyp_concat = f"{hyp_a} Also, {hyp_b}"

        judged = llm_client.judge_nli_label(prem, hyp_concat, n_retries=8)
        if judged is None:
            continue
        results.append({
            "premise": prem,
            "hypothesis": hyp_concat,
            "label": judged,
            "source": "concat"
        })
    # 若不足 n_items，可以继续随机补充尝试（略）。保持简单，交给上层用 max_trials 控制生成数量。
    return results


def short_simple_generate(llm_client, n_items: int, max_trials: int = 200) -> List[Dict]:
    return llm_client.generate_short_simple(n_items=n_items, max_trials=max_trials)


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
    llm_max_trials: int = 200,
) -> Tuple[Optional[List[int]], Optional[List[Dict]]]:
    """
    统一入口：
    - 返回 (indices, None) 表示从 pool 选中的索引
    - 返回 (None, new_samples) 表示策略直接“生成/判定”的新样本（不属于 pool）
    """
    s = strategy.lower()
    if s == "random":
        idxs = random_select(unlabeled_pool, budget_b, seed=seed)
        return idxs, None
    elif s == "entropy":
        idxs = entropy_select(
            unlabeled_pool, model, tokenizer, batch_size, max_len, budget_b,
            confidence_filter=confidence_filter
        )
        return idxs, None
    elif s in ("hyp_concat", "hyp-concat", "hypconcat"):
        assert llm_client is not None, "Hyp-Concat 需要 llm_client"
        new_samples = hyp_concat_generate(llm_client, unlabeled_pool, n_items=budget_b)
        return None, new_samples
    elif s in ("short_simple", "short&simple", "short-simple", "shortsimple"):
        assert llm_client is not None, "Short&Simple 需要 llm_client"
        new_samples = short_simple_generate(llm_client, n_items=budget_b, max_trials=llm_max_trials)
        return None, new_samples
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
