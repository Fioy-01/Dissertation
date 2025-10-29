# -*- coding: utf-8 -*-
"""
NLI 数据集定义（与 train_utils.build_loader 配套）：

- 两种样本格式均可：
  A) 原始文本样本：
     {
       "premise": "...",
       "hypothesis": "...",
       "label": "entailment" | "neutral" | "contradiction"   # 可选
       # 可选附加字段： "sample_weights": 1.0
     }

  B) 预处理样本（已tokenize）：
     {
       "input_ids": [int, int, ...],
       "attention_mask": [0/1, 0/1, ...],
       "labels": int (0/1/2)                                   # 可选
       # 可选附加字段： "sample_weights": 1.0
     }

- 数据集中不做 tokenize；一切交给 train_utils.build_loader 的 collate_fn 处理。
- 若样本不含 label，训练时会自动忽略监督信号（用于未标注池/伪标前阶段）。
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional

import json
import os
from torch.utils.data import Dataset

LABELS = ["entailment", "neutral", "contradiction"]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}
ID2LABEL = {i: l for l, i in LABEL2ID.items()}


def _norm_label_to_id(v: Any) -> Optional[int]:
    """将多种形式的标签转换为 id；未知/缺失返回 None。"""
    if v is None:
        return None
    if isinstance(v, int):
        return int(v) if int(v) in ID2LABEL else None
    s = str(v).strip().lower()
    return LABEL2ID.get(s, None)


def _looks_tokenized(sample: Dict[str, Any]) -> bool:
    """判断样本是否为预处理（已tokenize）格式。"""
    return ("input_ids" in sample) and ("attention_mask" in sample)


def _has_text_fields(sample: Dict[str, Any]) -> bool:
    return ("premise" in sample) and ("hypothesis" in sample)


class NliDataset(Dataset):
    """
    最小而健壮的 NLI 数据集封装。
    - 不在 __getitem__ 中进行 tokenize，避免与不同 tokenizer 绑定；
    - 直接返回样本（原始或已tokenize），由外层 collate 决定如何打包张量。
    """

    def __init__(self, rows: List[Dict[str, Any]], tokenizer=None, max_len: int = 256, strict: bool = False):
        """
        Args:
            rows: 样本列表（原始或预处理格式混合也可）
            tokenizer/max_len: 为了保持与旧接口兼容而保留；本类内部不使用
            strict: 若 True，遇到非法样本将抛出异常；否则尽量容错并跳过
        """
        super().__init__()
        self.rows: List[Dict[str, Any]] = []
        self.strict = bool(strict)
        bad = 0
        for i, r in enumerate(rows):
            ok, fixed = self._sanitize_one(r)
            if ok:
                self.rows.append(fixed)
            else:
                bad += 1
                if self.strict:
                    raise ValueError(f"NliDataset: invalid sample at index {i}: {r}")
        if bad > 0:
            print(f"[NliDataset] Skipped {bad} invalid samples; kept {len(self.rows)}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.rows[idx]

    # ----------------------------
    # 工具：清洗并标准化单条样本
    # ----------------------------
    def _sanitize_one(self, sample: Dict[str, Any]) -> (bool, Dict[str, Any]):
        if not isinstance(sample, dict):
            return False, {}

        s = dict(sample)  # 浅拷贝，避免原地修改

        # 情况A：已tokenize
        if _looks_tokenized(s):
            # labels -> 可选存在；如没有，尝试从 label 映射
            if "labels" not in s and "label" in s:
                lid = _norm_label_to_id(s["label"])
                if lid is not None:
                    s["labels"] = lid
            # sample_weights（可选）保留原值
            return True, s

        # 情况B：原始文本
        if _has_text_fields(s):
            # 统一类型
            s["premise"] = "" if s.get("premise") is None else str(s.get("premise"))
            s["hypothesis"] = "" if s.get("hypothesis") is None else str(s.get("hypothesis"))
            # 归一化 label（不强制要求存在）
            if "label" in s:
                lid = _norm_label_to_id(s["label"])
                if lid is not None:
                    # 同时保留原 label 字段（字符串）与 labels（id）
                    s["labels"] = lid
                    s["label"] = LABELS[lid]
                else:
                    # 未知标签时，删除 labels 键，保留原始 label 字段供上层决定
                    if "labels" in s:
                        s.pop("labels", None)
            # 保留 sample_weights（若提供）
            return True, s

        # 两种格式都不满足 → 非法样本
        return False, {}

    # ----------------------------
    # 便捷 I/O（可选）
    # ----------------------------
    @staticmethod
    def read_jsonl(path: str) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
        return rows

    @staticmethod
    def write_jsonl(path: str, rows: List[Dict[str, Any]]):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

