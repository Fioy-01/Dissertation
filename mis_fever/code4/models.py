import torch
import numpy as np
from typing import Dict, Optional
from torch.amp import autocast, GradScaler
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader
from torch.optim import AdamW
from sklearn.metrics import f1_score, precision_recall_fscore_support, accuracy_score
import torch.nn.functional as F


class Classifier:
    """
    统一分类器封装：
    - 支持标准训练 / 验证 / 概率预测 / 表示抽取
    - 训练端新增不均衡鲁棒化：Logit-Adjustment (LA) 与 Focal Loss（可并用）
    - 支持样本权 (batch["sample_weights"]) —— 用于方差/类稀缺自适应加权
    - 与 train_utils.train_with_early_stop 兼容（需提供 init_optimizer / init_scheduler / train_epoch / evaluate）
    """
    def __init__(self, model_dir: str, num_labels: int = 3, fp16: bool = True):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True, use_fast=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_dir,
            num_labels=num_labels,
            local_files_only=True
        ).to(self.device)
        self.fp16 = fp16 and (self.device.type == "cuda")

        # —— 类不均衡处理（可选开启）——
        # class_priors: 形如 [p(entailment), p(neutral), p(contradiction)]，总和=1
        # la_tau: Logit-Adjustment 温度系数 τ，e.g. 0.7
        # focal_gamma: Focal Loss γ，e.g. 1.5
        self.class_priors: Optional[np.ndarray] = None  # shape [num_labels]
        self.la_tau: Optional[float] = None
        self.focal_gamma: Optional[float] = None

    # ----------------------------
    # 配置不均衡鲁棒项
    # ----------------------------
    def set_imbalance_handles(
        self,
        class_priors: Optional[Dict[int, float]] = None,
        la_tau: Optional[float] = None,
        focal_gamma: Optional[float] = None
    ):
        """
        Args:
            class_priors: dict {class_id -> prior prob}, 会转成按 id 排列的数组
            la_tau: Logit-Adjustment 温度系数（None 表示关闭）
            focal_gamma: Focal Loss γ（None 表示关闭）
        """
        if class_priors is not None:
            # 依据模型 num_labels 构建数组
            K = self.model.config.num_labels
            arr = np.zeros((K,), dtype=np.float32)
            for k in range(K):
                arr[k] = float(class_priors.get(k, 1e-6))
            s = float(arr.sum())
            if s <= 0:
                # 兜底均匀先验
                arr[:] = 1.0 / K
            else:
                arr /= s
            self.class_priors = arr
        self.la_tau = la_tau
        self.focal_gamma = focal_gamma

    def get_tokenizer(self):
        return self.tokenizer

    # ----------------------------
    # 优化器 & 调度器（与 train_utils 对齐）
    # ----------------------------
    def init_optimizer(self, lr: float = 2e-5, weight_decay: float = 0.01):
        no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
        grouped = [
            {
                "params": [p for n, p in self.model.named_parameters()
                           if not any(nd in n for nd in no_decay)],
                "weight_decay": weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters()
                           if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        return AdamW(grouped, lr=lr)

    def init_scheduler(self, optimizer: AdamW, num_warmup_steps: int, num_training_steps: int):
        return get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )

    # ----------------------------
    # 训练一个 epoch（已集成 LA / Focal / 样本权）
    # ----------------------------
    def train_epoch(self, dataloader: DataLoader, optimizer: AdamW, scheduler=None) -> float:
        self.model.train()
        total_loss = 0.0
        scaler = GradScaler("cuda", enabled=self.fp16)

        # 预构建 LA 的偏置（常量向量）
        la_bias = None
        if self.la_tau is not None and self.class_priors is not None:
            la_bias = -float(self.la_tau) * torch.log(
                torch.tensor(self.class_priors, device=self.device, dtype=torch.float32).clamp_min(1e-8)
            )  # [C]

        for batch in dataloader:
            optimizer.zero_grad(set_to_none=True)
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)

            with autocast("cuda", enabled=self.fp16):
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits  # [B,C]

                # —— Logit-Adjustment —— (logits += adj)
                if la_bias is not None:
                    logits = logits + la_bias.to(logits.dtype)

                # —— 基础 CE（逐样本）——
                ce = F.cross_entropy(logits, labels, reduction="none")

                # —— Focal —— (ce *= (1 - p_t)^gamma)
                if self.focal_gamma is not None:
                    p_t = F.softmax(logits, dim=-1).gather(dim=-1, index=labels.view(-1, 1)).squeeze(1)
                    p_t = p_t.clamp_min(1e-8)
                    ce = ce * ((1.0 - p_t) ** float(self.focal_gamma))

                # —— 样本权（可选）——
                # 例如在 VaGeRy memorization 阶段或 reception 阶段对“高方差/少数类”样本加权
                # 通过 DataLoader 的 collate_fn 注入 batch["sample_weights"]
                sample_weights = batch.get("sample_weights", None)
                if sample_weights is not None:
                    sw = sample_weights.to(ce.device, dtype=ce.dtype)
                    loss = (ce * sw).mean()
                else:
                    loss = ce.mean()

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            if scheduler is not None:
                scheduler.step()
            total_loss += loss.detach().item()

        return total_loss / max(1, len(dataloader))

    # ----------------------------
    # 【修改】 验证评估方法
    # ----------------------------
    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        preds = []
        golds = []
        for batch in dataloader:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)

            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            pred = torch.argmax(logits, dim=-1)
            preds.extend(pred.detach().cpu().numpy().tolist())
            golds.extend(labels.detach().cpu().numpy().tolist())

        # 【修改】 计算并返回更详细的指标
        acc = accuracy_score(golds, preds)
        # 计算宏平均指标
        p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(golds, preds, average="macro", zero_division=0)
        # 计算每个类别的指标
        p_class, r_class, f1_class, s_class = precision_recall_fscore_support(golds, preds, labels=list(range(self.model.config.num_labels)), average=None, zero_division=0)

        return {
            "accuracy": float(acc),
            "macro_f1": float(f1_macro),
            "macro_precision": float(p_macro),
            "macro_recall": float(r_macro),
            "per_class_f1": f1_class.tolist(),
            "per_class_recall": r_class.tolist(),
            "per_class_precision": p_class.tolist(),
        }

    # ----------------------------
    # 概率预测（不修改 logits；用于采样/方差估计的基础）
    # ----------------------------
    @torch.no_grad()
    def predict_proba(self, dataloader: DataLoader) -> np.ndarray:
        """
        Return probability array [N, C].
        注意：这里使用模型原始输出的 softmax 概率，不注入 LA/Focal，
        以免在不同阶段（例如方差估计）引入额外干扰。
        """
        self.model.eval()
        probs = []
        for batch in dataloader:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            p = F.softmax(logits, dim=-1).detach().cpu().numpy()
            probs.append(p)
        return np.concatenate(probs, axis=0)

    # ----------------------------
    # VaGeRy 需要：返回 (p, z)
    # ----------------------------
    @torch.no_grad()
    def get_probs_and_repr(self, batch: Dict) -> tuple[torch.Tensor, torch.Tensor]:
        """
        返回 (p, z)，用于 VaGeRy：
        - p: softmax 概率 [B, C]
        - z: 最后一层 CLS 表示 [B, D]
        """
        self.model.eval()
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        outputs = self.model(input_ids=input_ids,
                             attention_mask=attention_mask,
                             output_hidden_states=True,
                             return_dict=True)
        logits = outputs.logits
        p = F.softmax(logits, dim=-1)
        last_hidden = outputs.hidden_states[-1]
        z = last_hidden[:, 0, :]
        return p, z

    # ----------------------------
    # 句向量抽取（采样/覆盖率/几何距离等用）
    # ----------------------------
    @torch.no_grad()
    def encode(self, dataloader: DataLoader, normalize: bool = False) -> np.ndarray:
        """
        Return sentence embeddings [N, D] using the last-layer CLS token.
        Works for BERT/RoBERTa-like backbones.

        normalize=True 可做 L2 归一化。
        """
        self.model.eval()
        vecs = []
        for batch in dataloader:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            outputs = self.model(input_ids=input_ids,
                                 attention_mask=attention_mask,
                                 output_hidden_states=True,
                                 return_dict=True)
            # 优先使用 pooler_output（若存在且非 None），否则用最后层 CLS
            z = None
            if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                z = outputs.pooler_output
            else:
                last_hidden = outputs.hidden_states[-1]
                z = last_hidden[:, 0, :]
            v = z.detach().cpu().numpy()
            vecs.append(v)
        X = np.concatenate(vecs, axis=0) if len(vecs) > 0 else np.zeros((0, self.model.config.hidden_size), dtype=np.float32)
        if normalize and X.shape[0] > 0:
            n = np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
            X = X / n
        return X
