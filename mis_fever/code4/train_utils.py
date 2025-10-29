# -*- coding: utf-8 -*-
import os
import random
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

# ------ 与工程其余部分的轻耦合常量 ------
LABELS = ["entailment", "neutral", "contradiction"]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}
ID2LABEL = {i: l for l, i in LABEL2ID.items()}


# =========================================================
# 随机种子
# =========================================================
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # cudnn 更稳定（如需极致速度可改为 benchmark=True）
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =========================================================
# DataLoader 构建
# - 适配两种数据路径：
#   1) 数据集 __getitem__ 已返回 tokenized 张量（含 input_ids/attention_mask[/labels]）
#   2) 数据集 __getitem__ 返回原始字段（premise/hypothesis[/label]），此处再做 tokenize
# - 可透传 sample_weights（若 __getitem__ 提供）
# =========================================================
def build_loader(
    dataset,
    batch_size: int,
    shuffle: bool,
    tokenizer,
    max_len: int,
    num_workers: int = 0,
    pin_memory: bool = True,
) -> DataLoader:
    """
    Args:
        dataset: 你的 NliDataset 或任意兼容 __len__/__getitem__ 的对象
        tokenizer: HF tokenizer（用于在 collate 中动态 tokenize；若样本已 tokenized 则不使用）
    Returns:
        torch.utils.data.DataLoader
    """
    def _to_tensor(x):
        if isinstance(x, torch.Tensor):
            return x
        return torch.tensor(x)

    def _label_to_id(v) -> Optional[int]:
        if v is None:
            return None
        if isinstance(v, int):
            return int(v)
        s = str(v).lower()
        return LABEL2ID.get(s, None)

    def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # 情况 A：样本已含 input_ids / attention_mask（可能还有 labels）
        has_tok = ("input_ids" in batch[0]) and ("attention_mask" in batch[0])

        if has_tok:
            input_ids = [_to_tensor(x["input_ids"]) for x in batch]
            attention_mask = [_to_tensor(x["attention_mask"]) for x in batch]
            out = {
                "input_ids": torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id),
                "attention_mask": torch.nn.utils.rnn.pad_sequence(attention_mask, batch_first=True, padding_value=0),
            }
            # labels（可选）
            if "labels" in batch[0]:
                labels = [_to_tensor(x["labels"]) for x in batch]
                out["labels"] = torch.stack(labels, dim=0)
            elif "label" in batch[0]:
                labels = [_label_to_id(x.get("label")) for x in batch]
                if all(l is not None for l in labels):
                    out["labels"] = torch.tensor(labels, dtype=torch.long)

            # 透传 sample_weights（若存在）
            if "sample_weights" in batch[0]:
                sw = [_to_tensor(x["sample_weights"]) for x in batch]
                out["sample_weights"] = torch.stack(sw, dim=0).float()

            return out

        # 情况 B：原始文本，需要在此处 tokenize
        premises = [str(x.get("premise", "")) for x in batch]
        hypos = [str(x.get("hypothesis", "")) for x in batch]
        enc = tokenizer(
            premises,
            hypos,
            padding=True,
            truncation=True,
            max_length=max_len,
            return_tensors="pt"
        )
        out = {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
        }

        # labels（可选）
        if "label" in batch[0]:
            labels = [_label_to_id(x.get("label")) for x in batch]
            if all(l is not None for l in labels):
                out["labels"] = torch.tensor(labels, dtype=torch.long)

        # sample_weights（可选，从样本字典中透传）
        if "sample_weights" in batch[0]:
            sw = [float(x.get("sample_weights", 1.0)) for x in batch]
            out["sample_weights"] = torch.tensor(sw, dtype=torch.float32)

        return out

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn
    )
    return loader


# =========================================================
# 【修改】 早停训练函数
# =========================================================
def train_with_early_stop(
    clf,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 3,
    patience: int = 2,
    lr: float = 2e-5,
    weight_decay: float = 0.01,
    warmup_ratio: float = 0.1,
) -> Dict[str, float]: # 【修改】返回值类型
    """
    Returns:
        best_val_metrics (Dict): The full metrics dictionary from the best epoch.
    """
    optimizer = clf.init_optimizer(lr=lr, weight_decay=weight_decay)
    total_steps = max(1, len(train_loader) * epochs)
    num_warmup = int(round(warmup_ratio * total_steps))
    scheduler = clf.init_scheduler(optimizer, num_warmup_steps=num_warmup, num_training_steps=total_steps)

    best_f1 = -1.0
    best_state = None
    best_metrics = {} # 【新增】用于保存最佳指标的完整字典
    wait = 0

    for ep in range(1, epochs + 1):
        train_loss = clf.train_epoch(train_loader, optimizer, scheduler)
        val_metrics = clf.evaluate(val_loader) # val_metrics 现在是包含所有指标的字典
        macro_f1 = float(val_metrics.get("macro_f1", 0.0))
        acc = float(val_metrics.get("accuracy", 0.0))

        print(f"[Epoch {ep:02d}] train_loss={train_loss:.6f}  "
              f"val_macro_f1={macro_f1:.6f}  val_acc={acc:.6f}")

        if macro_f1 > best_f1:
            best_f1 = macro_f1
            best_metrics = val_metrics # 【修改】保存整个字典
            # 保存最佳权重（CPU 拷贝，稳定）
            best_state = {k: v.detach().cpu() for k, v in clf.model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"Early stop triggered at epoch {ep}.")
                break

    if best_state is not None:
        clf.model.load_state_dict(best_state, strict=True)

    return best_metrics # 【修改】返回完整的最佳指标字典
