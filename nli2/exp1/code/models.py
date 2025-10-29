import torch
import numpy as np
from typing import Dict
from torch.amp import autocast, GradScaler
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader
from torch.optim import AdamW
from sklearn.metrics import f1_score, precision_recall_fscore_support, accuracy_score
import torch.nn.functional as F


class Classifier:
    def __init__(self, model_dir: str, num_labels: int = 3, fp16: bool = True):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True, use_fast=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_dir,
            num_labels=num_labels,
            local_files_only=True
        ).to(self.device)
        self.fp16 = fp16 and (self.device.type == "cuda")

    def get_tokenizer(self):
        return self.tokenizer

    def train_epoch(self, dataloader: DataLoader, optimizer: AdamW, scheduler=None) -> float:
        self.model.train()
        total_loss = 0.0
        scaler = GradScaler("cuda", enabled=self.fp16)
        for batch in dataloader:
            optimizer.zero_grad(set_to_none=True)
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)

            with autocast("cuda", enabled=self.fp16):
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            if scheduler is not None:
                scheduler.step()
            total_loss += loss.detach().item()
        return total_loss / max(1, len(dataloader))

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

        acc = accuracy_score(golds, preds)
        precision, recall, f1, _ = precision_recall_fscore_support(golds, preds, average="macro", zero_division=0)
        return {"accuracy": acc, "macro_f1": f1, "precision": precision, "recall": recall}

    @torch.no_grad()
    def predict_proba(self, dataloader: DataLoader) -> np.ndarray:
        """
        Return probability array [N, C].
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

    def init_optimizer(self, lr: float, weight_decay: float = 0.0):
        return AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)

    def init_scheduler(self, optimizer: AdamW, num_warmup_steps: int, num_training_steps: int):
        return get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)

