# -*- coding: utf-8 -*-
"""
VaGeRy minimal implementation:
- Rectifier（复用你已有 rectifier.py 中的思想，这里用同名类包装一下接口）
- VariationalHead：为 z 提供 (mu, logvar) 与 KL
- VaGeRyLossCfg：超参配置
- clone_teacher_from_student：EMA teacher 初始化
- memorization_phase：在无标注集上只训练 rectifier/vhead，计算诊断指标
- reception_phase：在有标注集上训练 student（用矫正后概率 p̄ 做 CE），并做 EMA 同步
- RectifiedModel：包装 predict_proba/encode，供采样阶段使用（返回的是 p̄）
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from datasets import NliDataset
from train_utils import build_loader


# -------------------------
# Rectifier（使用与你提供的思路一致的实现）
# -------------------------
class Rectifier(nn.Module):
    def __init__(self, dim_z: int, num_classes: int = 3, hidden_mul: float = 1.0, delta_scale: float = 1.0):
        super().__init__()
        h = int(dim_z * hidden_mul)
        self.mlp = nn.Sequential(
            nn.Linear(dim_z, h),
            nn.Tanh(),
            nn.Linear(h, num_classes)
        )
        self.delta_scale = float(delta_scale)
        self._last_kl: Optional[torch.Tensor] = None

    def forward(self, p: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        # p: [B,C] 概率；w: [B,D] 表示差
        delta = self.mlp(w) * self.delta_scale         # [B,C]
        logits_new = torch.log(p.clamp_min(1e-8)) + delta
        p_new = F.softmax(logits_new, dim=-1)
        # KL(p_new || p) 用于日志/轻正则
        self._last_kl = F.kl_div(
            p_new.log().clamp_min(-1e4),
            p,
            reduction="batchmean",
            log_target=False
        )
        return p_new

    def kl_loss(self) -> torch.Tensor:
        if self._last_kl is None:
            # 放在正确设备上
            return torch.tensor(0.0, device=next(self.parameters()).device)
        return self._last_kl

    @torch.no_grad()
    def delta_norm(self, w: torch.Tensor) -> torch.Tensor:
        delta = self.mlp(w) * self.delta_scale
        return delta.norm(dim=-1).mean()


# -------------------------
# Variational Head（轻量 KL 正则）
# -------------------------
class VariationalHead(nn.Module):
    def __init__(self, dim_z: int):
        super().__init__()
        self.fc_mu = nn.Linear(dim_z, dim_z)
        self.fc_logvar = nn.Linear(dim_z, dim_z)

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mu = self.fc_mu(z)
        logvar = self.fc_logvar(z).clamp(min=-10.0, max=10.0)
        return mu, logvar

    def kl(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        # KL(N(mu, sigma^2) || N(0, I))
        return 0.5 * torch.mean(torch.exp(logvar) + mu**2 - 1.0 - logvar)


# -------------------------
# 配置
# -------------------------
@dataclass
class VaGeRyLossCfg:
    epsilon_u: float = 0.05   # UCR 松弛
    epsilon_f: float = 0.03   # FR 松弛
    lambda_u: float = 1.0     # UCR 权重
    delta_f: float = 0.1      # FR  权重
    beta_v: float = 1e-4      # KL  权重


# -------------------------
# Teacher 初始化
# -------------------------
def clone_teacher_from_student(student) -> "Classifier":
    import copy
    # student 是你项目里的 Classifier 实例
    t = copy.deepcopy(student)
    t.model.eval()
    for p in t.model.parameters():
        p.requires_grad_(False)
    return t


# -------------------------
# 工具函数
# -------------------------
def _get_z_logits(model, batch, use_hidden_states=True):
    """
    返回 (p, z)：
      p: 概率 [B,C]
      z: 表示 [B,D] （取最后一层 CLS；若有 pooler_output 也可优先用）
    """
    outputs = model.model(
        input_ids=batch["input_ids"].to(model.device),
        attention_mask=batch["attention_mask"].to(model.device),
        output_hidden_states=True,
        return_dict=True
    )
    logits = outputs.logits
    p = F.softmax(logits, dim=-1)

    if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
        z = outputs.pooler_output
    else:
        # 取最后层 CLS
        z = outputs.hidden_states[-1][:, 0, :]
    return p, z


def _entropy(p: torch.Tensor) -> torch.Tensor:
    return -(p * (p.clamp_min(1e-8).log())).sum(dim=-1)


# -------------------------
# Memorization（只训 Rectifier + VHead）
# -------------------------
def memorization_phase(
    student,
    teacher,
    rectifier: Rectifier,
    vhead: VariationalHead,
    unlabeled_rows: List[Dict],
    batch_size: int,
    tokenizer,
    max_len: int,
    optim_rect: torch.optim.Optimizer,
    loss_cfg: VaGeRyLossCfg,
    max_steps: Optional[int] = None,
) -> Dict[str, float]:

    device = student.device
    rectifier.train()
    vhead.train()
    student.model.eval()
    teacher.model.eval()

    ds = NliDataset(unlabeled_rows, tokenizer=tokenizer, max_len=max_len)
    ld = build_loader(ds, batch_size=batch_size, shuffle=True, tokenizer=tokenizer, max_len=max_len)

    # 统计量
    Hs_all, Ht_all = [], []
    UCR_all, FR_all, KL_all, Dn_all = [], [], [], []

    step = 0
    for batch in ld:
        step += 1
        if max_steps is not None and step > max_steps:
            break

        with torch.no_grad():
            ps, zs = _get_z_logits(student, batch)
            pt, zt = _get_z_logits(teacher, batch)

        # 差向量（teacher - student）
        w = (zt - zs).detach()     # [B,D]
        w_t = (zs - zt).detach()

        # 矫正分布
        p_bar  = rectifier(ps, w)      # student 矫正
        p_bar_t = rectifier(pt, w_t)   # teacher 也过一次，作为参照

        # UCR：不确定性一致性
        H_s = _entropy(p_bar)          # [B]
        H_t = _entropy(p_bar_t)        # [B]
        ucr = torch.relu(H_t - H_s - loss_cfg.epsilon_u).mean()

        # FR：类级波动限制（类别维标准差）
        fr = torch.relu((p_bar - p_bar_t).std(dim=0).mean() - loss_cfg.epsilon_f)

        # 变分 KL（针对 z；给 rectifier 的输入做温和先验）
        mu, logvar = vhead(zs.detach())
        kl_v = vhead.kl(mu, logvar)

        loss = loss_cfg.lambda_u * ucr + loss_cfg.delta_f * fr + loss_cfg.beta_v * kl_v

        optim_rect.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(rectifier.parameters()) + list(vhead.parameters()), max_norm=5.0)
        optim_rect.step()

        # 统计
        with torch.no_grad():
            Hs_all.append(H_s.mean().item())
            Ht_all.append(H_t.mean().item())
            UCR_all.append(ucr.item())
            FR_all.append(fr.item())
            KL_all.append(rectifier.kl_loss().item())  # 使用 rectifier 内部 KL（p̄||p）
            Dn_all.append(rectifier.delta_norm(w).item())

    stats = {
        "mean_H_student_unlab": float(np.mean(Hs_all)) if Hs_all else 0.0,
        "mean_H_teacher_unlab": float(np.mean(Ht_all)) if Ht_all else 0.0,
        "mean_UCR_violation": float(np.mean(UCR_all)) if UCR_all else 0.0,
        "mean_FR_violation": float(np.mean(FR_all)) if FR_all else 0.0,
        "LV_kl": float(np.mean(KL_all)) if KL_all else 0.0,
        "rectify_delta_norm": float(np.mean(Dn_all)) if Dn_all else 0.0,
    }
    return stats


# -------------------------
# Reception（训练 student，EMA 同步 teacher）
# -------------------------
def reception_phase(
    student,
    teacher,
    rectifier: Rectifier,
    labeled_rows: List[Dict],
    val_rows: List[Dict],
    batch_size: int,
    tokenizer,
    max_len: int,
    epochs: int,
    patience: int,
    lr: float,
    ema_m: float = 0.99,
) -> Dict[str, float]:

    device = student.device
    student.model.train()
    rectifier.eval()
    teacher.model.eval()

    ds_tr = NliDataset(labeled_rows, tokenizer=tokenizer, max_len=max_len)
    ld_tr = build_loader(ds_tr, batch_size=batch_size, shuffle=True, tokenizer=tokenizer, max_len=max_len)

    # 简易优化器与调度（沿用你的 Classifier 的方式）
    optimizer = student.init_optimizer(lr=lr)

    best_val = -1.0
    wait = 0

    for ep in range(1, epochs + 1):
        # 训练一个 epoch
        for batch in ld_tr:
            student.model.train()

            outputs = student.model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                output_hidden_states=True,
                return_dict=True
            )
            logits = outputs.logits
            zs = outputs.hidden_states[-1][:, 0, :]

            with torch.no_grad():
                out_t = teacher.model(
                    input_ids=batch["input_ids"].to(device),
                    attention_mask=batch["attention_mask"].to(device),
                    output_hidden_states=True,
                    return_dict=True
                )
                zt = out_t.hidden_states[-1][:, 0, :]

            w = (zt - zs).detach()
            p = F.softmax(logits, dim=-1)
            p_bar = rectifier(p, w)

            # CE over rectified probs
            y = batch["labels"].to(device)
            loss = F.nll_loss(p_bar.clamp_min(1e-8).log(), y)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.model.parameters(), max_norm=5.0)
            optimizer.step()

            # EMA 同步 teacher
            with torch.no_grad():
                for tp, sp in zip(teacher.model.parameters(), student.model.parameters()):
                    tp.data = ema_m * tp.data + (1.0 - ema_m) * sp.data

        # 简易验证：用 student 原始输出在 val 上做 macro_f1（与你原有 eval 保持一致）
        # 如果更严格，也可在这里用 rectified 的分布评估。
        from sklearn.metrics import f1_score, accuracy_score
        ds_val = NliDataset(val_rows, tokenizer=tokenizer, max_len=max_len)
        ld_val = build_loader(ds_val, batch_size=batch_size, shuffle=False, tokenizer=tokenizer, max_len=max_len)
        student.model.eval()
        preds, golds = [], []
        with torch.no_grad():
            for b in ld_val:
                out = student.model(input_ids=b["input_ids"].to(device), attention_mask=b["attention_mask"].to(device))
                pred = out.logits.argmax(dim=-1)
                preds.extend(pred.cpu().tolist())
                golds.extend(b["labels"].tolist())
        # macro-f1
        from sklearn.metrics import precision_recall_fscore_support
        _p, _r, f1, _ = precision_recall_fscore_support(golds, preds, average="macro", zero_division=0)
        if f1 > best_val:
            best_val = f1
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    return {"val_macro_f1": float(best_val)}


# -------------------------
# 选样阶段用的包装器：返回 p̄
# -------------------------
class RectifiedModel:
    """
    供 sampling.py 使用的“模型”包装：
      - predict_proba(loader) 返回 p̄（矫正后的概率）
      - encode(loader) 直接使用 student.encode（和原逻辑一致）
    """
    def __init__(self, student, teacher, rectifier: Rectifier):
        self.student = student       # Classifier
        self.teacher = teacher       # Classifier
        self.rectifier = rectifier
        self.device = student.device
        self.tokenizer = student.get_tokenizer()

    def get_tokenizer(self):
        return self.tokenizer

    @torch.no_grad()
    def predict_proba(self, dataloader) -> np.ndarray:
        self.student.model.eval()
        self.teacher.model.eval()
        self.rectifier.eval()

        probs_all = []
        for batch in dataloader:
            # student
            out_s = self.student.model(
                input_ids=batch["input_ids"].to(self.device),
                attention_mask=batch["attention_mask"].to(self.device),
                output_hidden_states=True,
                return_dict=True
            )
            ps = F.softmax(out_s.logits, dim=-1)
            zs = out_s.hidden_states[-1][:, 0, :]

            # teacher
            out_t = self.teacher.model(
                input_ids=batch["input_ids"].to(self.device),
                attention_mask=batch["attention_mask"].to(self.device),
                output_hidden_states=True,
                return_dict=True
            )
            zt = out_t.hidden_states[-1][:, 0, :]

            w = (zt - zs)
            p_bar = self.rectifier(ps, w)
            probs_all.append(p_bar.detach().cpu().numpy())

        return np.concatenate(probs_all, axis=0)

    @torch.no_grad()
    def encode(self, dataloader) -> np.ndarray:
        # 与原 encode 保持一致：用 student 的 CLS 表示
        return self.student.encode(dataloader, normalize=False)

