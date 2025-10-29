# -*- coding: utf-8 -*-
"""
VaGeRy implementation (class- & variance-adaptive ready):
- Rectifier：接收差向量 w 与轻量上下文特征，输出 Δ，并通过 KL(p̄||p) 轻正则
- VariationalHead：为 z 提供 (mu, logvar) 与 KL
- VaGeRyLossCfg：UCR/FR/LV 各项权重
- AdaptiveCfg：基于“类稀缺 + 方差”的自适应阈值与样本权
- clone_teacher_from_student：EMA teacher 初始化
- memorization_phase：在无标注集上训练 rectifier/vhead，并输出统计指标
- reception_phase：在有标注集上训练 student（常规 CE），并做 EMA 同步
- RectifiedModel：封装 predict_proba/encode，使调用方得到 rectified 概率（p̄）
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from datasets import NliDataset
from train_utils import build_loader


# -------------------------
# 小工具
# -------------------------
class LayerNorm1D(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.ln = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ln(x)


def _entropy(p: torch.Tensor) -> torch.Tensor:
    """Batch-wise entropy: H(p) = -sum p log p, 返回 [B]"""
    p = p.clamp_min(1e-8)
    return -(p * p.log()).sum(dim=-1)


# -------------------------
# Rectifier（含 LayerNorm + tanh 限幅 + 特征拼接）
# -------------------------
class Rectifier(nn.Module):
    """
    输入:
      - p:    原始预测分布 [B, C]
      - w:    hidden state 差向量（teacher - student）[B, D]
      - v_hat:归一化方差分数 [B] （0..1）
      - Hs,Ht: student/teacher 的熵 [B]

    输出:
      - p_new: rectified 概率分布 [B, C]

    轻正则:
      - KL(p_new || p) 记入 self._last_kl（batchmean）
    """
    def __init__(self, dim_z: int, num_classes: int = 3, hidden_mul: float = 1.0, delta_scale: float = 1.0):
        super().__init__()
        h = int(dim_z * hidden_mul)
        self.norm = LayerNorm1D(dim_z)
        self.mlp = nn.Sequential(
            nn.Linear(dim_z + 3, h),   # +3: [v_hat, Hs, Ht]
            nn.Tanh(),
            nn.Linear(h, num_classes),
            nn.Tanh()                  # 轻限幅，防止早期抖动
        )
        self.delta_scale = float(delta_scale)
        self._last_kl: Optional[torch.Tensor] = None

    def forward(
        self,
        p: torch.Tensor,
        w: torch.Tensor,
        v_hat: torch.Tensor,
        Hs: torch.Tensor,
        Ht: torch.Tensor
    ) -> torch.Tensor:
        wn = self.norm(w)  # [B,D]
        feats = torch.cat([wn, v_hat.unsqueeze(1), Hs.unsqueeze(1), Ht.unsqueeze(1)], dim=1)  # [B, D+3]
        delta = self.mlp(feats) * self.delta_scale  # [B,C]
        logits_new = torch.log(p.clamp_min(1e-8)) + delta
        p_new = F.softmax(logits_new, dim=-1)
        # KL(p_new || p)，batchmean -> 标量
        self._last_kl = F.kl_div(
            p_new.log().clamp_min(-1e4),
            p,
            reduction="batchmean",
            log_target=False
        )
        return p_new

    def kl_loss(self) -> torch.Tensor:
        if self._last_kl is None:
            return torch.tensor(0.0, device=next(self.parameters()).device)
        return self._last_kl

    @torch.no_grad()
    def delta_norm(self, w: torch.Tensor) -> torch.Tensor:
        zero_ctx = torch.zeros((w.size(0), 3), device=w.device, dtype=w.dtype)
        delta = self.mlp(torch.cat([self.norm(w), zero_ctx], dim=1))
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
        # KL(N(mu, sigma^2) || N(0, I)) -> 标量
        return 0.5 * torch.mean(torch.exp(logvar) + mu**2 - 1.0 - logvar)


# -------------------------
# 配置
# -------------------------
@dataclass
class VaGeRyLossCfg:
    epsilon_u: float = 0.05   # UCR 松弛（若未启用自适应，可直接使用）
    epsilon_f: float = 0.03   # FR  松弛
    lambda_u: float = 1.0     # UCR 权重
    delta_f: float = 0.1      # FR  权重
    beta_v: float = 1e-4      # 变分 KL 权重


@dataclass
class AdaptiveCfg:
    # 阈值基线
    eps_u0: float = 0.06
    eps_f0: float = 0.03
    # 方差影响系数
    alpha_u: float = 0.5
    alpha_f: float = 0.5
    # 类稀缺影响系数
    beta_u: float = 0.5
    beta_f: float = 0.5
    # 样本权：方差权 & 类权
    gamma_v: float = 1.0
    lambda_cls: float = 0.5


def _adaptive_thresholds_weights(
    v_hat: torch.Tensor,     # [B] in [0,1]
    pi_hat: torch.Tensor,    # [B] in [0,1]  类稀缺度的“反向”(这里用 π_k = N_k / max_j N_j，后面使用 (1-π_k) )
    acfg: AdaptiveCfg,
    device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """返回 (eps_u, eps_f, w_all) 逐样本向量"""
    v_hat = v_hat.to(device)
    pi_hat = pi_hat.to(device)

    eps_u = acfg.eps_u0 * (1.0 - acfg.alpha_u * v_hat) * (1.0 - acfg.beta_u * (1.0 - pi_hat))
    eps_f = acfg.eps_f0 * (1.0 - acfg.alpha_f * v_hat) * (1.0 - acfg.beta_f * (1.0 - pi_hat))

    w_var = 1.0 + acfg.gamma_v * v_hat
    mean1m = torch.clamp(1.0 - pi_hat, min=1e-8).mean()
    w_cls = 1.0 + acfg.lambda_cls * (1.0 - pi_hat) / torch.clamp(mean1m, min=1e-8)
    w_all = (w_var * w_cls).to(device)
    return eps_u, eps_f, w_all


# -------------------------
# Teacher 初始化（EMA 冻结参数）
# -------------------------
def clone_teacher_from_student(student) -> "Classifier":
    import copy
    t = copy.deepcopy(student)
    t.model.eval()
    for p in t.model.parameters():
        p.requires_grad_(False)
    return t


# -------------------------
# 内部工具：取 (p,z)
# -------------------------
def _get_z_logits(model, batch) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    返回 (p, z):
      p: 概率 [B,C]
      z: 表示 [B,D]（优先 pooler_output，否则最后层 CLS）
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
        z = outputs.hidden_states[-1][:, 0, :]
    return p, z


# -------------------------
# Memorization（只训练 rectifier/vhead）
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
    acfg: Optional[AdaptiveCfg] = None,
    class_scarcity_getter: Optional[Callable[[List[int]], List[float]]] = None,
    warmup_only_lv: bool = False,  # True 则仅训练变分 KL（前1-2轮）
) -> Dict[str, float]:

    device = student.device
    rectifier.train()
    vhead.train()
    student.model.eval()
    teacher.model.eval()

    if acfg is None:
        acfg = AdaptiveCfg()

    ds = NliDataset(unlabeled_rows, tokenizer=tokenizer, max_len=max_len)
    ld = build_loader(ds, batch_size=batch_size, shuffle=True, tokenizer=tokenizer, max_len=max_len)

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

        # 近似 v_hat：用 student 的熵做 min-max 归一化（开销低、稳定）
        with torch.no_grad():
            Hs_raw = _entropy(ps)              # [B]
            Ht_raw = _entropy(pt)              # [B]
            mn, mx = Hs_raw.min(), Hs_raw.max()
            denom = torch.clamp(mx - mn, min=1e-6)
            v_hat = (Hs_raw - mn) / denom      # [B] in [0,1]

            # 类稀缺度 π_k：由外部提供函数（基于预测类）；若缺省，则退化为全 1
            if class_scarcity_getter is None:
                pi_hat = torch.ones_like(v_hat, device=v_hat.device)
            else:
                y_pred = ps.argmax(dim=-1).detach().cpu().tolist()  # int ids
                pi_arr = class_scarcity_getter(y_pred)               # list of floats in [0,1]
                pi_hat = torch.tensor(pi_arr, device=v_hat.device, dtype=v_hat.dtype)

        # 矫正前后：student 修正、teacher 也过一遍作参照
        p_bar   = rectifier(ps, zt - zs, v_hat, Hs_raw, Ht_raw)
        kl_s    = rectifier.kl_loss().detach()  # 第一遍 KL（如需统计）
        p_bar_t = rectifier(pt, zs - zt, v_hat, Ht_raw, Hs_raw)
        # kl_t  = rectifier.kl_loss().detach()  # 如需双向 KL，可启用

        # 自适应阈值与样本权
        eps_u, eps_f, w_all = _adaptive_thresholds_weights(v_hat.detach(), pi_hat.detach(), acfg, device)

        # 熵（rectified 后）
        H_s = _entropy(p_bar)     # [B]
        H_t = _entropy(p_bar_t)   # [B]

        if warmup_only_lv:
            # 仅训练变分 KL（针对 z）
            mu, logvar = vhead(zs.detach())
            kl_v = vhead.kl(mu, logvar)                 # 标量
            loss = loss_cfg.beta_v * kl_v
        else:
            # UCR（逐样本 -> mean）
            ucr = torch.relu(H_t - H_s - eps_u).mean()  # 标量

            # FR：batch 级分歧标量 - 标量阈值
            fr_measure = (p_bar - p_bar_t).std(dim=0).mean()  # 标量
            fr = torch.relu(fr_measure - eps_f.mean())        # 标量

            # 变分 KL：标量
            mu, logvar = vhead(zs.detach())
            kl_v = vhead.kl(mu, logvar)                       # 标量

            # 样本权放在批外以均值形式调节项权重
            w_mean = w_all.mean()
            loss = (
                loss_cfg.lambda_u * (ucr * w_mean)
                + loss_cfg.delta_f * (fr * w_mean)
                + loss_cfg.beta_v * kl_v
            )

        optim_rect.zero_grad(set_to_none=True)
        # 开发期可打开安全检查
        # assert loss.dim() == 0, f"loss must be scalar, got {tuple(loss.shape)}"
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(rectifier.parameters()) + list(vhead.parameters()), max_norm=5.0)
        optim_rect.step()

        # 统计
        with torch.no_grad():
            Hs_all.append(H_s.mean().item())
            Ht_all.append(H_t.mean().item())
            if not warmup_only_lv:
                UCR_all.append(torch.relu(H_t - H_s - eps_u).mean().item())
                FR_all.append(torch.relu(fr_measure - eps_f.mean()).item())
            # 统计 KL：这里默认记录 student->rectified 的 KL
            KL_all.append(kl_s.item())
            Dn_all.append(rectifier.delta_norm(zt - zs).item())

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
# Reception（监督训练 student，EMA 同步 teacher）
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
    lr: float = 2e-5,
    epochs: int = 3,
    patience: int = 2,
    ema_m: float = 0.99,
    use_sample_weights: bool = False,
    sample_weight_fn: Optional[Callable[[Dict], torch.Tensor]] = None,
):
    """
    说明：
      - 为保持稳定与低计算开销，此处 Reception 使用常规 CE 训练 student；
      - rectifier 在该阶段不更新（更稳更省），仅参与推理时的 p̄（由 RectifiedModel 使用）；
      - 训练后进行一次 EMA 同步：teacher = m*teacher + (1-m)*student。
    """
    device = student.device
    # 构造 DataLoader
    ds_train = NliDataset(labeled_rows, tokenizer=tokenizer, max_len=max_len)
    ld_train = build_loader(ds_train, batch_size=batch_size, shuffle=True, tokenizer=tokenizer, max_len=max_len)
    ds_val = NliDataset(val_rows, tokenizer=tokenizer, max_len=max_len)
    ld_val = build_loader(ds_val, batch_size=batch_size, shuffle=False, tokenizer=tokenizer, max_len=max_len)

    optimizer = student.init_optimizer(lr=lr)
    total_steps_hint = len(ld_train) * epochs
    scheduler = student.init_scheduler(
        optimizer,
        num_warmup_steps=max(0, int(0.1 * total_steps_hint)),
        num_training_steps=total_steps_hint
    )

    best_metric = -1.0
    best_state = None
    wait = 0
    scaler = torch.amp.GradScaler("cuda", enabled=student.fp16)

    for ep in range(1, epochs + 1):
        # --- train ---
        student.model.train()
        total_loss = 0.0
        for batch in ld_train:
            optimizer.zero_grad(set_to_none=True)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            with torch.amp.autocast("cuda", enabled=student.fp16):
                outputs = student.model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                loss = F.cross_entropy(logits, labels, reduction="none")
                if use_sample_weights and sample_weight_fn is not None:
                    sw = sample_weight_fn(batch).to(loss.device, dtype=loss.dtype)
                    loss = (loss * sw).mean()
                else:
                    loss = loss.mean()

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            if scheduler is not None:
                scheduler.step()
            total_loss += loss.detach().item()

        # --- validate ---
        student.model.eval()
        preds, golds = [], []
        with torch.no_grad():
            for batch in ld_val:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                outputs = student.model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                p = torch.argmax(logits, dim=-1)
                preds.extend(p.detach().cpu().numpy().tolist())
                golds.extend(labels.detach().cpu().numpy().tolist())

        from sklearn.metrics import accuracy_score, precision_recall_fscore_support
        acc = accuracy_score(golds, preds)
        prec, rec, f1, _ = precision_recall_fscore_support(golds, preds, average="macro", zero_division=0)
        metric = float(f1)

        if metric > best_metric:
            best_metric = metric
            best_state = {k: v.detach().cpu() for k, v in student.model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    if best_state is not None:
        student.model.load_state_dict(best_state)

    # --- EMA 同步 teacher 参数 ---
    with torch.no_grad():
        m = float(ema_m)
        for (name_s, ps), (name_t, pt) in zip(student.model.named_parameters(), teacher.model.named_parameters()):
            if ps.data.shape == pt.data.shape:
                pt.data.mul_(m).add_((1.0 - m) * ps.data)
    teacher.model.eval()


# -------------------------
# RectifiedModel（用于选样阶段返回 rectified 概率）
# -------------------------
class RectifiedModel:
    """
    用法：
      rm = RectifiedModel(student, teacher, rectifier)
      probs = rm.predict_proba(dataloader)  # 返回 rectified 概率 p̄

    说明：
      - 这里按 batch 内做 v_hat（在 Hs 的 min-max 上归一化）
      - 若需要更稳定的 v_hat（跨全池），可先扫一遍统计后再第二次前向；为了低开销，此处采用单遍。
    """
    def __init__(self, student, teacher, rectifier: Rectifier):
        self.student = student
        self.teacher = teacher
        self.rectifier = rectifier
        self.device = student.device

    @torch.no_grad()
    def predict_proba(self, dataloader) -> np.ndarray:
        self.student.model.eval()
        self.teacher.model.eval()
        self.rectifier.eval()
        probs = []

        for batch in dataloader:
            ps, zs = _get_z_logits(self.student, batch)
            pt, zt = _get_z_logits(self.teacher, batch)
            Hs_raw = _entropy(ps)
            Ht_raw = _entropy(pt)
            # batch 内归一化
            mn, mx = Hs_raw.min(), Hs_raw.max()
            denom = torch.clamp(mx - mn, min=1e-6)
            v_hat = (Hs_raw - mn) / denom
            p_bar = self.rectifier(ps, zt - zs, v_hat, Hs_raw, Ht_raw)
            probs.append(p_bar.detach().cpu().numpy())
        return np.concatenate(probs, axis=0)

    @torch.no_grad()
    def encode(self, dataloader, normalize: bool = False) -> np.ndarray:
        """透传 student 的 encode（选样/覆盖率用）"""
        return self.student.encode(dataloader, normalize=normalize)


