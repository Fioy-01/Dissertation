import os
import csv
import random
import numpy as np
import torch
from typing import Dict, List, Optional
from torch.utils.data import DataLoader


def set_seed(seed: int = 42):
    """
    固定随机种子，保证实验可复现
    """
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
    支持有标签和无标签的数据集（无标签时不返回 labels 张量）。
    """
    def collate_fn(features: List[Dict]):
        premises = [f["premise"] for f in features]
        hypos = [f["hypothesis"] for f in features]

        enc = tokenizer(
            premises,
            hypos,
            padding=True,
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
        )

        # ★ 可选标签（无标签时不返回）
        if "labels" in features[0]:
            labels = torch.tensor([f["labels"] for f in features], dtype=torch.long)
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
    """
    训练模型并在验证集上进行早停。
    - clf: 你的 Classifier 实例
    - train_loader / val_loader: DataLoader
    - patience: 宏平均 F1 在验证集连续多少轮不提升就停止
    """
    optimizer = clf.init_optimizer(lr=lr)
    if total_steps_hint is None:
        total_steps_hint = len(train_loader) * epochs
    scheduler = clf.init_scheduler(
        optimizer,
        num_warmup_steps=max(0, int(0.1 * total_steps_hint)),
        num_training_steps=total_steps_hint
    )

    best_metric = -1.0
    best_state = None
    wait = 0

    for ep in range(1, epochs + 1):
        train_loss = clf.train_epoch(train_loader, optimizer, scheduler)
        metrics = clf.evaluate(val_loader)
        macro_f1 = metrics["macro_f1"]

        print(f"[Epoch {ep}/{epochs}] train_loss={train_loss:.4f} "
              f"val_macro_f1={macro_f1:.4f} acc={metrics['accuracy']:.4f}")

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

def memorization_phase(student, teacher, rectifier, unlabeled_loader, optimizer, eps_u=0.05, eps_f=0.03, beta=1e-4):
    """
    仅更新 rectifier（memorization 阶段）
    """
    rectifier.train()
    student.eval()
    teacher.eval()

    for batch in unlabeled_loader:
        with torch.no_grad():
            p, z = student.get_probs_and_repr(batch)
            p_star, z_star = teacher.get_probs_and_repr(batch)

        w = z_star - z
        w_star = z - z_star

        p_bar, delta = rectifier(p, w)
        p_bar_star, _ = rectifier(p_star, w_star)

        # UCR
        H = -(p_bar * torch.log(torch.clamp(p_bar, 1e-8, 1))).sum(dim=-1).mean()
        H_star = -(p_bar_star * torch.log(torch.clamp(p_bar_star, 1e-8, 1))).sum(dim=-1).mean()
        LUCR = torch.relu(H_star - H - eps_u).mean()

        # FR
        diff = p_bar - p_bar_star
        std_c = diff.std(dim=0).mean()
        LFR = torch.relu(std_c - eps_f)

        # LV
        KL = rectifier.kl_reg(z)

        loss = LUCR + 0.1 * LFR + beta * KL

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def reception_phase(student, teacher, rectifier, labeled_loader, optimizer, ema_m=0.99):
    """
    更新 student（reception 阶段）
    """
    student.train()
    rectifier.eval()  # 通常固定
    teacher.eval()

    for batch in labeled_loader:
        p, z = student.get_probs_and_repr(batch)
        with torch.no_grad():
            _, z_star = teacher.get_probs_and_repr(batch)

        w = z_star - z
        p_bar, _ = rectifier(p, w)

        labels = batch["labels"].to(p.device)
        loss = torch.nn.functional.cross_entropy(p_bar, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # EMA 更新 teacher
        for param, t_param in zip(student.parameters(), teacher.parameters()):
            t_param.data.mul_(ema_m).add_(param.data, alpha=1 - ema_m)


