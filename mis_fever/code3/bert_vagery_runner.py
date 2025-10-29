# Write the runner file to /mnt/data without importing project modules in this environment.
import os
import json
import math
import time
import random
from copy import deepcopy
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict, Counter

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from datasets import load_jsonl, save_jsonl, NliDataset, LABEL2ID, ID2LABEL
from models import Classifier
from train_utils import set_seed, build_loader, train_with_early_stop
from sampling import select as al_select


# ----------------------------
# Utilities
# ----------------------------

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def timestamp():
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

ROUND_LOG_HEADER = [
    "timestamp","strategy","mode","round","labeled_total","B",
    "minority_class","minority_f1","minority_recall","minority_precision",
    "macro_f1","accuracy","macro_precision","macro_recall",
    "pi_e","pi_n","pi_c",
    "n_e","n_n","n_c",
    "kappa_uniform","kappa_invfreq","rho_t_max","rho_t_p95",
    "gen_G_used","gen_pass","gen_pass_rate",
    # VaGeRy diagnostics (6)
    "mean_H_student_unlab","mean_H_teacher_unlab",
    "mean_UCR_violation","mean_FR_violation",
    "LV_kl","rectify_delta_norm"
]

def log_round_metrics(round_log_path, data_dict):
    new_file = not os.path.exists(round_log_path)
    with open(round_log_path, "a", encoding="utf-8") as fw:
        if new_file:
            fw.write(",".join(ROUND_LOG_HEADER) + "\n")
        fw.write(",".join(map(str, [
            timestamp(),
            data_dict["strat"],
            data_dict["mode"],
            data_dict["round"],
            data_dict["labeled_total"],
            data_dict["B"],
            data_dict["minority_class"],
            f"{data_dict['minority_f1']:.4f}",
            f"{data_dict['minority_recall']:.4f}",
            f"{data_dict['minority_precision']:.4f}",
            f"{data_dict['macro_f1']:.4f}",
            f"{data_dict['accuracy']:.4f}",
            f"{data_dict['macro_precision']:.4f}",
            f"{data_dict['macro_recall']:.4f}",
            f"{data_dict['pi_e']:.4f}",
            f"{data_dict['pi_n']:.4f}",
            f"{data_dict['pi_c']:.4f}",
            data_dict["n_e"],
            data_dict["n_n"],
            data_dict["n_c"],
            f"{data_dict['kappa_uniform']:.6f}",
            f"{data_dict['kappa_invfreq']:.6f}",
            f"{data_dict['rho_t_max']:.6f}",
            f"{data_dict['rho_t_p95']:.6f}",
            data_dict.get("gen_G_used", 0),
            data_dict.get("gen_pass", 0),
            f"{data_dict.get('gen_pass_rate', 0.0):.4f}",
            # 6 diagnostics
            f"{data_dict.get('mean_H_student_unlab',0.0):.6f}",
            f"{data_dict.get('mean_H_teacher_unlab',0.0):.6f}",
            f"{data_dict.get('mean_UCR_violation',0.0):.6f}",
            f"{data_dict.get('mean_FR_violation',0.0):.6f}",
            f"{data_dict.get('LV_kl',0.0):.6f}",
            f"{data_dict.get('rectify_delta_norm',0.0):.6f}",
        ])) + "\n")


def strat_name_clean(s: str) -> str:
    return s.replace("/", "-").replace("&", "_").replace(" ", "").lower()


def load_split(init_dir: str):
    L = load_jsonl(os.path.join(init_dir, "init_labeled.jsonl"))
    U = load_jsonl(os.path.join(init_dir, "unlabeled_pool.jsonl"))
    return L, U

def load_eval_sets(val_path: str, test_path: str):
    V = load_jsonl(val_path)
    T = load_jsonl(test_path) if (test_path and os.path.exists(test_path)) else []
    return V, T


def build_dataset_and_loader(rows: List[Dict], tokenizer, batch_size: int, max_len: int, shuffle: bool):
    ds = NliDataset(rows, tokenizer=tokenizer, max_len=max_len)
    return ds, build_loader(ds, batch_size=batch_size, shuffle=shuffle, tokenizer=tokenizer, max_len=max_len)


def compute_per_class_metrics(golds: List[int], preds: List[int], n_classes: int = 3):
    from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix
    acc = accuracy_score(golds, preds)
    prec, rec, f1, support = precision_recall_fscore_support(golds, preds, labels=list(range(n_classes)), average=None, zero_division=0)
    macro_prec = float(np.mean(prec))
    macro_rec = float(np.mean(rec))
    macro_f1 = float(np.mean(f1))
    cm = confusion_matrix(golds, preds, labels=list(range(n_classes)))
    return {
        "accuracy": float(acc),
        "macro_precision": macro_prec,
        "macro_recall": macro_rec,
        "macro_f1": float(macro_f1),
        "per_class_precision": prec.tolist(),
        "per_class_recall": rec.tolist(),
        "per_class_f1": f1.tolist(),
        "support": support.tolist(),
        "confusion_matrix": cm.tolist(),
    }

def eval_model_on_split(clf: Classifier, rows: List[Dict], batch_size: int, max_len: int):
    tok = clf.get_tokenizer()
    _, loader = build_dataset_and_loader(rows, tok, batch_size=batch_size, max_len=max_len, shuffle=False)
    clf.model.eval()
    preds, golds = [], []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(clf.device)
            attention_mask = batch["attention_mask"].to(clf.device)
            labels = batch["labels"].to(clf.device)
            out = clf.model(input_ids=input_ids, attention_mask=attention_mask)
            p = torch.argmax(out.logits, dim=-1)
            preds.extend(p.detach().cpu().numpy().tolist())
            golds.extend(labels.detach().cpu().numpy().tolist())
    return compute_per_class_metrics(golds, preds, n_classes=3)


class EnsembleModel:
    def __init__(self, members: List[Classifier]):
        assert len(members) > 0
        self.members = members
        self.tokenizer = members[0].get_tokenizer()
        self.device = members[0].device

    def predict_proba(self, dataloader) -> np.ndarray:
        probs_list = []
        for m in self.members:
            p = m.predict_proba(dataloader)
            probs_list.append(p)
        return np.mean(np.stack(probs_list, axis=0), axis=0)

    def encode(self, dataloader) -> np.ndarray:
        return self.members[0].encode(dataloader, normalize=False)

    def get_tokenizer(self):
        return self.tokenizer


def majority_minority_classes(counts: Dict[str, int]):
    if not counts:
        return [], []
    min_cnt = min(counts.values())
    max_cnt = max(counts.values())
    mins = [k for k, v in counts.items() if v == min_cnt]
    maxs = [k for k, v in counts.items() if v == max_cnt]
    return mins, maxs


def count_by_label(rows: List[Dict]) -> Dict[str, int]:
    cnt = {"entailment": 0, "neutral": 0, "contradiction": 0}
    for r in rows:
        lab = r.get("label")
        if isinstance(lab, int):
            lab = ID2LABEL[lab]
        if lab in cnt:
            cnt[lab] += 1
    return cnt


def compute_coverage_metrics(labeled_rows: List[Dict], unlabeled_rows: List[Dict], ref_model: Classifier, batch_size: int, max_len: int):
    tok = ref_model.get_tokenizer()
    _, ld_L = build_dataset_and_loader(labeled_rows, tok, batch_size=batch_size, max_len=max_len, shuffle=False)
    emb_L = ref_model.encode(ld_L, normalize=False)
    if len(unlabeled_rows) == 0 or emb_L.shape[0] == 0:
        return {"rho_t_max": 0.0, "rho_t_p95": 0.0}
    sample_U = unlabeled_rows
    if len(sample_U) > 5000:
        sample_idx = np.random.choice(len(sample_U), size=5000, replace=False).tolist()
        sample_U = [unlabeled_rows[i] for i in sample_idx]
    _, ld_U = build_dataset_and_loader(sample_U, tok, batch_size=batch_size, max_len=max_len, shuffle=False)
    emb_U = ref_model.encode(ld_U, normalize=False)
    S = emb_L.astype(np.float32)
    U = emb_U.astype(np.float32)
    S_sq = (S ** 2).sum(axis=1)
    dmins = []
    chunk = 1024
    for i in range(0, U.shape[0], chunk):
        Uc = U[i:i+chunk]
        U_sq = (Uc ** 2).sum(axis=1, keepdims=True)
        dot = Uc @ S.T
        d2 = U_sq + S_sq[None, :] - 2.0 * dot
        dmin = np.sqrt(np.maximum(0.0, d2.min(axis=1)))
        dmins.append(dmin)
    dmins = np.concatenate(dmins, axis=0)
    rho_max = float(np.max(dmins))
    rho_p95 = float(np.percentile(dmins, 95))
    return {"rho_t_max": rho_max, "rho_t_p95": rho_p95}


# ----------------------------
# Local VaGeRy (with requested options)
# ----------------------------

@dataclass
class LossCfg:
    epsilon_u: float = 0.05
    epsilon_f: float = 0.03
    lambda_u: float = 1.0
    delta_f: float = 0.1
    beta_v: float = 1e-4
    enable_ucr: bool = True
    enable_fr: bool = True

class VariationalHead(nn.Module):
    def __init__(self, dim_z: int):
        super().__init__()
        self.fc_mu = nn.Linear(dim_z, dim_z)
        self.fc_logvar = nn.Linear(dim_z, dim_z)

    def forward(self, z):
        mu = self.fc_mu(z)
        logvar = self.fc_logvar(z).clamp(min=-10.0, max=10.0)
        return mu, logvar

    def kl(self, mu, logvar):
        return 0.5 * torch.mean(torch.exp(logvar) + mu**2 - 1.0 - logvar)

class Rectifier(nn.Module):
    def __init__(self, dim_in: int, num_classes: int = 3,
                 hidden_mul: float = 1.0, delta_scale: float = 1.0,
                 target: str = "logp", temperature: float = 1.0):
        super().__init__()
        h = int(dim_in * hidden_mul)
        self.mlp = nn.Sequential(
            nn.Linear(dim_in, h),
            nn.Tanh(),
            nn.Linear(h, num_classes)
        )
        self.delta_scale = float(delta_scale)
        self._last_kl: Optional[torch.Tensor] = None
        self.target = target
        self.temperature = float(temperature)

    def forward(self, *, p: Optional[torch.Tensor], logits: Optional[torch.Tensor], w: torch.Tensor):
        delta = self.mlp(w) * self.delta_scale
        if self.target == "logits":
            assert logits is not None
            logits_new = logits + delta
            T = max(1e-8, self.temperature)
            p_new = torch.softmax(logits_new / T, dim=-1)
            with torch.no_grad():
                p_ref = torch.softmax(logits / T, dim=-1)
            self._last_kl = F.kl_div(p_new.log().clamp_min(-1e-4), p_ref, reduction="batchmean", log_target=False)
        else:
            assert p is not None
            logits_new = p.clamp_min(1e-8).log() + delta
            T = max(1e-8, self.temperature)
            p_new = torch.softmax(logits_new / T, dim=-1)
            self._last_kl = F.kl_div(p_new.log().clamp_min(-1e-4), p, reduction="batchmean", log_target=False)
        return p_new

    def kl_loss(self):
        if self._last_kl is None:
            return torch.tensor(0.0, device=next(self.parameters()).device)
        return self._last_kl

    @torch.no_grad()
    def delta_norm(self, w):
        delta = self.mlp(w) * self.delta_scale
        return delta.norm(dim=-1).mean()

@torch.no_grad()
def get_p_z_logits(model: Classifier, batch):
    outputs = model.model(
        input_ids=batch["input_ids"].to(model.device),
        attention_mask=batch["attention_mask"].to(model.device),
        output_hidden_states=True,
        return_dict=True
    )
    logits = outputs.logits
    p = torch.softmax(logits, dim=-1)
    z = outputs.hidden_states[-1][:, 0, :]
    return p, z, logits

def compute_class_prototypes(model: Classifier, labeled_rows, tokenizer, batch_size, max_len, num_classes=3):
    ds = NliDataset(labeled_rows, tokenizer=tokenizer, max_len=max_len)
    ld = build_loader(ds, batch_size=batch_size, shuffle=False, tokenizer=tokenizer, max_len=max_len)
    buckets = defaultdict(list)
    model.model.eval()
    with torch.no_grad():
        for batch in ld:
            outputs = model.model(
                input_ids=batch["input_ids"].to(model.device),
                attention_mask=batch["attention_mask"].to(model.device),
                output_hidden_states=True,
                return_dict=True
            )
            z = outputs.hidden_states[-1][:, 0, :]
            labs = batch["labels"].to(model.device)
            for i in range(z.size(0)):
                buckets[int(labs[i].item())].append(z[i])
    protos = {}
    for k in range(num_classes):
        if len(buckets[k]) == 0:
            raise RuntimeError("No samples to compute prototype for class {}".format(k))
        else:
            protos[k] = torch.stack(buckets[k], dim=0).mean(dim=0)
    return protos

def memorization_phase(student: Classifier, teacher: Classifier, rectifier: Rectifier, vhead: VariationalHead,
                       unlabeled_rows, batch_size, tokenizer, max_len,
                       optim_rect, loss_cfg: LossCfg, max_steps=None,
                       w_mode: str = "diff", delta_target: Optional[str] = None, temperature: Optional[float] = None,
                       labeled_rows=None, proto_source: str = "teacher"):
    device = student.device
    rectifier.train(); vhead.train()
    student.model.eval(); teacher.model.eval()

    if delta_target is not None:
        rectifier.target = delta_target
    if temperature is not None:
        rectifier.temperature = float(temperature)

    protos = None
    if w_mode.startswith("proto"):
        assert labeled_rows is not None, "proto_* requires labeled_rows"
        base = teacher if (proto_source == "teacher") else student
        protos = compute_class_prototypes(base, labeled_rows, tokenizer, batch_size, max_len)

    ds = NliDataset(unlabeled_rows, tokenizer=tokenizer, max_len=max_len)
    ld = build_loader(ds, batch_size=batch_size, shuffle=True, tokenizer=tokenizer, max_len=max_len)

    Hs_all, Ht_all, UCR_all, FR_all, KL_all, Dn_all = [], [], [], [], [], []
    step = 0
    for batch in ld:
        step += 1
        if max_steps is not None and step > max_steps:
            break

        with torch.no_grad():
            ps, zs, ls = get_p_z_logits(student, batch)
            pt, zt, lt = get_p_z_logits(teacher, batch)

        if w_mode == "diff":
            w = (zt - zs)
            w_t = (zs - zt)
        elif w_mode == "concat":
            w = torch.cat([zt - zs, zs, zt], dim=-1)
            w_t = torch.cat([zs - zt, zt, zs], dim=-1)
        elif w_mode in ("proto_t", "proto_s"):
            assert protos is not None
            y_hat = pt.argmax(dim=-1) if (w_mode == "proto_t") else ps.argmax(dim=-1)
            c = torch.stack([protos[int(k.item())] for k in y_hat], dim=0).to(device)
            w = (c - zs)
            w_t = (zs - c)
        else:
            raise ValueError(f"Unknown w_mode: {w_mode}")

        p_bar  = rectifier(p=ps, logits=ls, w=w)
        p_bar_t = rectifier(p=pt, logits=lt, w=w_t)

        H_s = -(p_bar  * (p_bar.clamp_min(1e-8).log())).sum(dim=-1)
        H_t = -(p_bar_t * (p_bar_t.clamp_min(1e-8).log())).sum(dim=-1)
        ucr = torch.relu(H_t - H_s - loss_cfg.epsilon_u).mean() if loss_cfg.enable_ucr else torch.zeros((), device=device)
        fr  = torch.relu((p_bar - p_bar_t).std(dim=0).mean() - loss_cfg.epsilon_f) if loss_cfg.enable_fr else torch.zeros((), device=device)

        mu, logvar = vhead(zs.detach()); kl_v = vhead.kl(mu, logvar)

        loss = loss_cfg.lambda_u * ucr + loss_cfg.delta_f * fr + loss_cfg.beta_v * kl_v

        optim_rect.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(rectifier.parameters()) + list(vhead.parameters()), max_norm=5.0)
        optim_rect.step()

        with torch.no_grad():
            Hs_all.append(H_s.mean().item())
            Ht_all.append(H_t.mean().item())
            UCR_all.append(ucr.item())
            FR_all.append(fr.item())
            KL_all.append(rectifier.kl_loss().item())
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

def reception_phase(student: Classifier, teacher: Classifier, rectifier: Rectifier,
                    labeled_rows, val_rows, batch_size, tokenizer, max_len,
                    epochs, patience, lr, ema_m=0.99,
                    w_mode: str = "diff", delta_target: Optional[str] = None, temperature: Optional[float] = None,
                    proto_source: str = "teacher"):
    if delta_target is not None:
        rectifier.target = delta_target
    if temperature is not None:
        rectifier.temperature = float(temperature)

    protos = None
    if w_mode.startswith("proto"):
        base = teacher if (proto_source == "teacher") else student
        protos = compute_class_prototypes(base, labeled_rows, tokenizer, batch_size, max_len)

    ds_tr = NliDataset(labeled_rows, tokenizer=tokenizer, max_len=max_len)
    ld_tr = build_loader(ds_tr, batch_size=batch_size, shuffle=True, tokenizer=tokenizer, max_len=max_len)

    opt = torch.optim.AdamW(student.model.parameters(), lr=lr)

    best_metric = -1.0
    best_state = None
    wait = 0

    ds_val = NliDataset(val_rows, tokenizer=tokenizer, max_len=max_len)
    ld_val = build_loader(ds_val, batch_size=batch_size, shuffle=False, tokenizer=tokenizer, max_len=max_len)

    for ep in range(1, epochs + 1):
        student.model.train(); rectifier.eval(); teacher.model.eval()
        for batch in ld_tr:
            ps, zs, ls = get_p_z_logits(student, batch)
            with torch.no_grad():
                pt, zt, lt = get_p_z_logits(teacher, batch)

            if w_mode == "diff":
                w = (zt - zs)
            elif w_mode == "concat":
                w = torch.cat([zt - zs, zs, zt], dim=-1)
            elif w_mode in ("proto_t", "proto_s"):
                assert protos is not None
                y_hat = pt.argmax(dim=-1) if (w_mode == "proto_t") else ps.argmax(dim=-1)
                c = torch.stack([protos[int(k.item())] for k in y_hat], dim=0).to(student.device)
                w = (c - zs)
            else:
                raise ValueError(f"Unknown w_mode: {w_mode}")

            p_bar = rectifier(p=ps, logits=ls, w=w)
            loss = F.cross_entropy(p_bar, batch["labels"].to(student.device))

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.model.parameters(), max_norm=1.0)
            opt.step()

            with torch.no_grad():
                for sp, tp in zip(student.model.parameters(), teacher.model.parameters()):
                    tp.data.mul_(ema_m).add_(sp.data * (1.0 - ema_m))

        # eval
        student.model.eval()
        preds, golds = [], []
        with torch.no_grad():
            for batch in ld_val:
                ps, zs, ls = get_p_z_logits(student, batch)
                if w_mode == "diff":
                    with torch.no_grad():
                        pt, zt, lt = get_p_z_logits(teacher, batch)
                    w = (zt - zs)
                elif w_mode == "concat":
                    with torch.no_grad():
                        pt, zt, lt = get_p_z_logits(teacher, batch)
                    w = torch.cat([zt - zs, zs, zt], dim=-1)
                elif w_mode in ("proto_t", "proto_s"):
                    base = teacher if (w_mode == "proto_t") else student
                    if protos is None:
                        protos = compute_class_prototypes(base, labeled_rows, tokenizer, batch_size, max_len)
                    y_hat = ps.argmax(dim=-1) if (w_mode == "proto_s") else get_p_z_logits(teacher, batch)[0].argmax(dim=-1)
                    c = torch.stack([protos[int(k.item())] for k in y_hat], dim=0).to(student.device)
                    w = (c - zs)
                else:
                    raise ValueError(f"Unknown w_mode: {w_mode}")
                p_bar = rectifier(p=ps, logits=ls, w=w)
                pred = torch.argmax(p_bar, dim=-1)
                preds.extend(pred.detach().cpu().numpy().tolist())
                golds.extend(batch["labels"].detach().cpu().numpy().tolist())

        from sklearn.metrics import f1_score
        macro_f1 = f1_score(golds, preds, average="macro", zero_division=0.0)
        if macro_f1 > best_metric:
            best_metric = macro_f1
            best_state = {k: v.detach().cpu() for k, v in student.model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    if best_state is not None:
        student.model.load_state_dict(best_state)
    return {"best_val_macro_f1": float(best_metric)}


# ----------------------------
# Runner main
# ----------------------------

def run(args):
    INIT_DIR = args.init_dir
    VAL_PATH = args.val_path
    TEST_PATH = args.test_path
    MODEL_DIR = args.model_dir

    base_out = args.output_root
    ensure_dir(base_out)

    init_L, init_U = load_split(INIT_DIR)
    val_rows, test_rows = load_eval_sets(VAL_PATH, TEST_PATH)

    temp_clf = Classifier(MODEL_DIR, num_labels=3, fp16=True)
    tokenizer = temp_clf.get_tokenizer()
    del temp_clf

    strategies = [s.strip() for s in args.strategies.split(",") if s.strip()]
    strategies = [strat_name_clean(s) for s in strategies]

    for strat in strategies:
        for mode in args.modes:
            mode = mode.lower()

            exp_name = f"BERT_ratio-real_{strat}_{mode}_w-{args.w_mode}_d-{args.delta_target}_T{args.temperature}_abl-{args.ablation}"
            out_dir = os.path.join(base_out, exp_name)
            ensure_dir(out_dir)
            ensure_dir(os.path.join(out_dir, "plots"))
            with open(os.path.join(out_dir, "run_config.json"), "w", encoding="utf-8") as f:
                json.dump({
                    "timestamp": timestamp(),
                    "strategy": strat,
                    "mode": mode,
                    "init_dir": INIT_DIR,
                    "val_path": VAL_PATH,
                    "test_path": TEST_PATH,
                    "model_dir": MODEL_DIR,
                    "rounds": args.rounds,
                    "budget_per_round": args.budget,
                    "ensemble_size": args.ensemble_size,
                    "batch_size": args.batch_size,
                    "max_len": args.max_len,
                    "epochs": args.epochs,
                    "patience": args.patience,
                    "lr": args.lr,
                    "w_mode": args.w_mode,
                    "delta_target": args.delta_target,
                    "temperature": args.temperature,
                    "ablation": args.ablation,
                }, f, ensure_ascii=False, indent=2)

            L = deepcopy(init_L)
            U = deepcopy(init_U)

            round_log_path = os.path.join(out_dir, "round_log.csv")

            for t in range(1, args.rounds + 1):
                print(f"\n=== [{exp_name}] Round {t}/{args.rounds} ===")

                # ---- Train ensemble from scratch on current L ----
                members: List[Classifier] = []
                seed_list = [args.seed + i for i in range(args.ensemble_size)]
                for s in seed_list:
                    set_seed(s)
                    clf = Classifier(MODEL_DIR, num_labels=3, fp16=True)
                    _, train_loader = build_dataset_and_loader(L, clf.get_tokenizer(), batch_size=args.batch_size, max_len=args.max_len, shuffle=True)
                    _, val_loader = build_dataset_and_loader(val_rows, clf.get_tokenizer(), batch_size=args.batch_size, max_len=args.max_len, shuffle=False)
                    train_with_early_stop(
                        clf, train_loader, val_loader,
                        epochs=args.epochs, patience=args.patience, lr=args.lr,
                        total_steps_hint=len(train_loader) * args.epochs
                    )
                    members.append(clf)

                # Evaluate ensemble members on val
                member_metrics = []
                for i, clf in enumerate(members):
                    m = eval_model_on_split(clf, val_rows, batch_size=args.batch_size, max_len=args.max_len)
                    m["seed"] = seed_list[i]
                    member_metrics.append(m)
                macro_f1s = [m["macro_f1"] for m in member_metrics]
                best_idx = int(np.argmax(macro_f1s))
                best_clf = members[best_idx]
                best_seed = seed_list[best_idx]

                ensemble = EnsembleModel(members)
                tok = ensemble.get_tokenizer()

                def meanstd(key):
                    arr = np.array([m[key] for m in member_metrics], dtype=np.float32)
                    return float(arr.mean()), float(arr.std())

                macro_f1_mean, macro_f1_std = meanstd("macro_f1")
                acc_mean, acc_std = meanstd("accuracy")
                macro_prec_mean, _ = meanstd("macro_precision")
                macro_rec_mean, _ = meanstd("macro_recall")

                per_class_f1_mean = np.mean(np.stack([m["per_class_f1"] for m in member_metrics], axis=0), axis=0).tolist()
                per_class_rec_mean = np.mean(np.stack([m["per_class_recall"] for m in member_metrics], axis=0), axis=0).tolist()
                per_class_prec_mean = np.mean(np.stack([m["per_class_precision"] for m in member_metrics], axis=0), axis=0).tolist()

                # ---- VaGeRy: build student/teacher/rectifier; run phases ----
                student = best_clf
                import copy
                teacher = copy.deepcopy(student)
                teacher.model.eval()
                for p in teacher.model.parameters():
                    p.requires_grad_(False)

                dim_z = student.model.config.hidden_size
                dim_in = dim_z if args.w_mode in ("diff","proto_t","proto_s") else (dim_z * 3)
                rectifier = Rectifier(dim_in=dim_in, num_classes=3, hidden_mul=1.0, delta_scale=1.0,
                                      target=args.delta_target, temperature=args.temperature).to(student.device)
                vhead = VariationalHead(dim_z=dim_z).to(student.device)
                optim_rect = torch.optim.AdamW(list(rectifier.parameters()) + list(vhead.parameters()), lr=3e-4)

                loss_cfg = LossCfg(
                    epsilon_u=0.05, epsilon_f=0.03, lambda_u=1.0, delta_f=0.1, beta_v=1e-4,
                    enable_ucr=(args.ablation in ("all","ucr")),
                    enable_fr=(args.ablation in ("all","fr"))
                )

                mem_size = min(len(U), 2000)
                U_mem = U if mem_size == len(U) else random.sample(U, mem_size)
                mem_stats = memorization_phase(
                    student=student, teacher=teacher, rectifier=rectifier, vhead=vhead,
                    unlabeled_rows=U_mem, batch_size=args.batch_size, tokenizer=tok, max_len=args.max_len,
                    optim_rect=optim_rect, loss_cfg=loss_cfg, max_steps=None,
                    w_mode=args.w_mode, delta_target=args.delta_target, temperature=args.temperature,
                    labeled_rows=L, proto_source=("teacher" if args.w_mode=="proto_t" else "student")
                )

                rec_metrics = reception_phase(
                    student=student, teacher=teacher, rectifier=rectifier,
                    labeled_rows=L, val_rows=val_rows, batch_size=args.batch_size, tokenizer=tok, max_len=args.max_len,
                    epochs=args.epochs, patience=args.patience, lr=args.lr, ema_m=0.99,
                    w_mode=args.w_mode, delta_target=args.delta_target, temperature=args.temperature,
                    proto_source=("teacher" if args.w_mode=="proto_t" else "student")
                )

                # ---- Selection on pool using ensemble ----
                ds_pool = NliDataset(U, tokenizer=tok, max_len=args.max_len)
                ld_pool = build_loader(ds_pool, batch_size=args.batch_size, shuffle=False, tokenizer=tok, max_len=args.max_len)

                # >>>>>>>>>>>> FIXED CALL SIGNATURE HERE <<<<<<<<<<<<
                selected_indices, new_samples = al_select(
                    strategy=strat,
                    unlabeled_pool=U,
                    model=ensemble,
                    tokenizer=tok,
                    batch_size=args.batch_size,
                    max_len=args.max_len,
                    budget_b=args.budget,
                    seed=args.seed,
                    labeled_set=L
                )
                # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

                S_new = [U[i] for i in selected_indices]
                cnt = Counter([ID2ID(x.get("label")) if isinstance(x.get("label"), int) else x.get("label","neutral") for x in S_new])
                n_e = int(cnt.get("entailment", 0)); n_n = int(cnt.get("neutral", 0)); n_c = int(cnt.get("contradiction", 0))
                total_sel = max(1, len(S_new))
                pi_e = n_e / total_sel; pi_n = n_n / total_sel; pi_c = n_c / total_sel

                cov = compute_coverage_metrics(L, U, best_clf, batch_size=args.batch_size, max_len=args.max_len)

                cnt_L = count_by_label(L)
                mins, _ = majority_minority_classes(cnt_L)
                minority = mins[0] if mins else "neutral"
                min_idx = LABEL2ID[minority]
                minority_f1 = float(per_class_f1_mean[min_idx])
                minority_rec = float(per_class_rec_mean[min_idx])
                minority_prec = float(per_class_prec_mean[min_idx])

                kappa_uniform = (pi_e + pi_n + pi_c) / 3.0
                total_L = sum(cnt_L.values()) if sum(cnt_L.values()) > 0 else 1
                inv_w = {
                    "entailment": (1.0 / max(1, cnt_L.get("entailment", 0))) if cnt_L.get("entailment",0)>0 else 0.0,
                    "neutral": (1.0 / max(1, cnt_L.get("neutral", 0))) if cnt_L.get("neutral",0)>0 else 0.0,
                    "contradiction": (1.0 / max(1, cnt_L.get("contradiction", 0))) if cnt_L.get("contradiction",0)>0 else 0.0,
                }
                denom = sum(inv_w.values()) if sum(inv_w.values())>0 else 1.0
                kappa_invfreq = (pi_e*inv_w["entailment"] + pi_n*inv_w["neutral"] + pi_c*inv_w["contradiction"]) / denom

                data_dict = {
                    "strat": strat, "mode": mode, "round": t,
                    "labeled_total": len(L), "B": args.budget,
                    "minority_class": minority,
                    "minority_f1": minority_f1, "minority_recall": minority_rec, "minority_precision": minority_prec,
                    "macro_f1": macro_f1_mean, "accuracy": acc_mean, "macro_precision": macro_prec_mean, "macro_recall": macro_rec_mean,
                    "pi_e": pi_e, "pi_n": pi_n, "pi_c": pi_c,
                    "n_e": n_e, "n_n": n_n, "n_c": n_c,
                    "kappa_uniform": kappa_uniform, "kappa_invfreq": kappa_invfreq,
                    "rho_t_max": cov["rho_t_max"], "rho_t_p95": cov["rho_t_p95"],
                    "gen_G_used": 0, "gen_pass": 0, "gen_pass_rate": 0.0,
                    # VaGeRy diag
                    "mean_H_student_unlab": mem_stats.get("mean_H_student_unlab", 0.0),
                    "mean_H_teacher_unlab": mem_stats.get("mean_H_teacher_unlab", 0.0),
                    "mean_UCR_violation": mem_stats.get("mean_UCR_violation", 0.0),
                    "mean_FR_violation": mem_stats.get("mean_FR_violation", 0.0),
                    "LV_kl": mem_stats.get("LV_kl", 0.0),
                    "rectify_delta_norm": mem_stats.get("rectify_delta_norm", 0.0),
                }
                log_round_metrics(round_log_path, data_dict)

                # update L / U
                L.extend(S_new)
                mask = np.ones(len(U), dtype=bool)
                mask[selected_indices] = False
                U = [u for i, u in enumerate(U) if mask[i]]

    print("Done.")


def ID2ID(x):
    if isinstance(x, int):
        return ID2LABEL[x]
    return x


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="VaGeRy+BERT runner with w/delta/ablation options")

    # Paths
    parser.add_argument("--init_dir", type=str, required=True, help="Directory containing init_labeled.jsonl and unlabeled_pool.jsonl")
    parser.add_argument("--val_path", type=str, required=True, help="Validation jsonl path")
    parser.add_argument("--test_path", type=str, required=False, default="", help="(Optional) Test jsonl path")
    parser.add_argument("--model_dir", type=str, required=True, help="HF local model directory")
    parser.add_argument("--output_root", type=str, default="outputs", help="Root output directory")

    # Experiment
    parser.add_argument("--strategies", type=str, default="alvin", help="Comma-separated strategies (e.g., alvin,entropy,random)")
    parser.add_argument("--modes", type=str, nargs="+", default=["controlled-optimal"], help="uncontrolled / controlled-balanced / controlled-optimal")
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--budget", type=int, default=150)
    parser.add_argument("--ensemble_size", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_len", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=42)

    # VaGeRy options
    parser.add_argument("--w_mode", type=str, default="diff",
                        choices=["diff","concat","proto_t","proto_s"],
                        help="w construction: diff=z*-z; concat=[z*-z;z;z*]; proto_* uses class prototypes")
    parser.add_argument("--delta_target", type=str, default="logp",
                        choices=["logp","logits"], help="Apply Δ to logp or logits")
    parser.add_argument("--temperature", type=float, default=1.0, help="Softmax temperature in rectified head")
    parser.add_argument("--ablation", type=str, default="all",
                        choices=["all","ucr","fr"], help="Ablation switch: only UCR, only FR, or both")

    args = parser.parse_args()
    run(args)

