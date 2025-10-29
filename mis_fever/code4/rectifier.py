import torch
import torch.nn as nn
import torch.nn.functional as F


class Rectifier(nn.Module):
    """
    VaGeRy 中的矫正器：
    - 输入: 学生/教师的 hidden 差 w
    - 输出: Δ 向量，用于修正概率分布
    - KL 正则：约束 rectified 分布与原分布的差异
    """

    def __init__(self, hidden_size: int, num_labels: int = 3, alpha: float = 1.0):
        """
        hidden_size: BERT/RoBERTa 隐层维度 (768 for base, 1024 for large)
        num_labels: 分类类别数
        alpha: KL 正则权重
        """
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, num_labels)
        )
        self.alpha = alpha
        self.last_kl = None  # 缓存最新 KL
        self.num_labels = num_labels

    def forward(self, p: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """
        p: 原始预测分布 [B, C]
        w: hidden state 差向量 [B, H]
        return: rectified 概率分布 [B, C]
        """
        delta = self.mlp(w)  # [B, C]
        logits_new = torch.log(p.clamp_min(1e-8)) + delta
        p_new = F.softmax(logits_new, dim=-1)

        # KL(p_new || p)
        kl = F.kl_div(
            p_new.log().clamp_min(-1e4),  # 避免 -inf
            p,
            reduction="batchmean",
            log_target=False
        )
        self.last_kl = kl
        return p_new

    def kl_loss(self) -> torch.Tensor:
        """返回最新 KL，用于日志记录"""
        if self.last_kl is None:
            return torch.tensor(0.0, device=next(self.parameters()).device)
        return self.last_kl

    def delta_norm(self, w: torch.Tensor) -> torch.Tensor:
        """计算 Δ(w) 的平均 L2 范数"""
        with torch.no_grad():
            delta = self.mlp(w)
            return delta.norm(dim=-1).mean()

