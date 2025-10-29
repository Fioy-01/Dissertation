import os
import csv
import random
import numpy as np
import torch
from typing import Dict, List, Optional
from torch.utils.data import DataLoader


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_loader(dataset, batch_size: int, shuffle: bool, tokenizer=None, max_len: int = 256):
    """
    使用 fast tokenizer 的 batch __call__ 在 collate_fn 里做批量分词与 padding。
    """
    def collate_fn(features: List[Dict]):
        premises = [f["premise"] for f in features]
        hypos = [f["hypothesis"] for f in features]
        labels = torch.tensor([f["labels"] for f in features], dtype=torch.long)

        enc = tokenizer(
            premises,
            hypos,
            padding=True,
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
        )
        enc["labels"] = labels
        return enc

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)


def train_with_early_stop(
    clf,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 3,
    patience: int = 2,
    lr: float = 2e-5,
    total_steps_hint: Optional[int] = None,
):
    optimizer = clf.init_optimizer(lr=lr)
    if total_steps_hint is None:
        total_steps_hint = len(train_loader) * epochs
    scheduler = clf.init_scheduler(optimizer, num_warmup_steps=max(0, int(0.1 * total_steps_hint)), num_training_steps=total_steps_hint)

    best_metric = -1.0
    best_state = None
    wait = 0

    for ep in range(1, epochs + 1):
        train_loss = clf.train_epoch(train_loader, optimizer, scheduler)
        metrics = clf.evaluate(val_loader)
        macro_f1 = metrics["macro_f1"]

        print(f"[Epoch {ep}/{epochs}] train_loss={train_loss:.4f} val_macro_f1={macro_f1:.4f} acc={metrics['accuracy']:.4f}")

        if macro_f1 > best_metric:
            best_metric = macro_f1
            best_state = {k: v.cpu() for k, v in clf.model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"Early stop triggered at epoch {ep}.")
                break

    if best_state is not None:
        clf.model.load_state_dict(best_state)
    return best_metric

