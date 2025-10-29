# --- 在文件顶部新增工具：类原型计算 ---
from collections import defaultdict
from datasets import NliDataset
from train_utils import build_loader

def compute_class_prototypes(model, labeled_rows, tokenizer, batch_size, max_len, num_classes=3):
    """
    用“有标注集”构建每类原型 c_k（CLS 向量均值）。
    返回：{0: torch.Tensor[D], 1: ..., 2: ...}（在 model.device 上）
    """
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
            # 取 CLS 表示
            z = outputs.hidden_states[-1][:, 0, :]  # [B, D]
            labs = batch["labels"].to(model.device) # [B]
            for i in range(z.size(0)):
                buckets[int(labs[i].item())].append(z[i])

    protos = {}
    for k in range(num_classes):
        if len(buckets[k]) == 0:
            # 没有该类样本时，退化为 0 向量
            protos[k] = torch.zeros_like(next(model.model.parameters())).new_zeros(
                (outputs.hidden_states[-1].size(-1),)
            )
        else:
            protos[k] = torch.stack(buckets[k], dim=0).mean(dim=0)
    return protos


# --- Rectifier：支持 target='logp'|'logits' 与温度 T；输入维度可变（diff/concat） ---
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
        self.target = target  # 'logp' or 'logits'
        self.temperature = float(temperature)

    def forward(self, *, p: Optional[torch.Tensor], logits: Optional[torch.Tensor], w: torch.Tensor) -> torch.Tensor:
        """
        p: 概率 [B,C]（当 target='logp' 使用）
        logits: 原始 logits [B,C]（当 target='logits' 使用）
        w: 输入向量（diff/concat/proto）[B, D_in]
        return p_bar: 矫正后的概率 [B,C]
        """
        delta = self.mlp(w) * self.delta_scale  # [B,C]
        if self.target == "logits":
            assert logits is not None, "target=logits 需要原始 logits"
            logits_new = logits + delta
            T = max(1e-8, self.temperature)
            p_new = torch.softmax(logits_new / T, dim=-1)
            # KL(p_new || softmax(logits/T))
            with torch.no_grad():
                p_ref = torch.softmax(logits / T, dim=-1)
            self._last_kl = F.kl_div(
                p_new.log().clamp_min(-1e4), p_ref, reduction="batchmean", log_target=False
            )
        else:
            assert p is not None, "target=logp 需要概率 p"
            logits_new = p.clamp_min(1e-8).log() + delta
            T = max(1e-8, self.temperature)
            p_new = torch.softmax(logits_new / T, dim=-1)
            # KL(p_new || p)
            self._last_kl = F.kl_div(
                p_new.log().clamp_min(-1e-4), p, reduction="batchmean", log_target=False
            )
        return p_new

    def kl_loss(self) -> torch.Tensor:
        if self._last_kl is None:
            return torch.tensor(0.0, device=next(self.parameters()).device)
        return self._last_kl

    @torch.no_grad()
    def delta_norm(self, w: torch.Tensor) -> torch.Tensor:
        delta = self.mlp(w) * self.delta_scale
        return delta.norm(dim=-1).mean()


# --- VaGeRy 配置：加入消融开关 ---
@dataclass
class VaGeRyLossCfg:
    epsilon_u: float = 0.05
    epsilon_f: float = 0.03
    lambda_u: float = 1.0  # UCR 权重
    delta_f: float = 0.1   # FR  权重
    beta_v: float = 1e-4   # 变分 KL 辅助项
    enable_ucr: bool = True
    enable_fr: bool = True


# --- 获取 p/z/logits 的工具 ---
def _get_p_z_logits(model, batch):
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


# --- Memorization：加入 w_mode / delta_target / temperature / 原型 ---
def memorization_phase(
    student, teacher, rectifier, vhead,
    unlabeled_rows, batch_size, tokenizer, max_len,
    optim_rect, loss_cfg: VaGeRyLossCfg,
    max_steps: Optional[int] = None,
    w_mode: str = "diff",                 # 'diff' | 'concat' | 'proto_t' | 'proto_s'
    delta_target: Optional[str] = None,   # 'logp' | 'logits' | None(=rectifier.target)
    temperature: Optional[float] = None,
    labeled_rows: Optional[List[Dict]] = None,  # proto 模式可用
    proto_source: str = "teacher"         # 'teacher' | 'student'
) -> Dict[str, float]:

    device = student.device
    rectifier.train(); vhead.train()
    student.model.eval(); teacher.model.eval()

    if delta_target is not None:
        rectifier.target = delta_target
    if temperature is not None:
        rectifier.temperature = float(temperature)

    # 可选：原型
    protos = None
    if w_mode.startswith("proto"):
        assert labeled_rows is not None, "proto_* 模式需要 labeled_rows 构建类原型"
        base = teacher if (proto_source == "teacher") else student
        protos = compute_class_prototypes(base, labeled_rows, tokenizer, batch_size, max_len)  # {k: z_k}

    ds = NliDataset(unlabeled_rows, tokenizer=tokenizer, max_len=max_len)
    ld = build_loader(ds, batch_size=batch_size, shuffle=True, tokenizer=tokenizer, max_len=max_len)

    Hs_all, Ht_all, UCR_all, FR_all, KL_all, Dn_all = [], [], [], [], [], []
    step = 0
    for batch in ld:
        step += 1
        if max_steps is not None and step > max_steps:
            break

        with torch.no_grad():
            ps, zs, ls = _get_p_z_logits(student, batch)
            pt, zt, lt = _get_p_z_logits(teacher, batch)

        # --- 构造 w ---
        if w_mode == "diff":
            w = (zt - zs)
            w_t = (zs - zt)
        elif w_mode == "concat":
            w = torch.cat([zt - zs, zs, zt], dim=-1)
            w_t = torch.cat([zs - zt, zt, zs], dim=-1)
        elif w_mode in ("proto_t", "proto_s"):
            assert protos is not None
            # 用 top-1 伪标签选择类原型
            if w_mode == "proto_t":
                y_hat = pt.argmax(dim=-1)  # [B]
            else:
                y_hat = ps.argmax(dim=-1)
            c = torch.stack([protos[int(k.item())] for k in y_hat], dim=0).to(device)  # [B,D]
            # 用 c 替代 z*：w = c - z
            w = (c - zs)
            # 对 teacher 分支做对称
            w_t = (zs - c)
        else:
            raise ValueError(f"Unknown w_mode: {w_mode}")

        # --- 矫正 ---
        p_bar  = rectifier(p=ps, logits=ls, w=w)    # student 矫正
        p_bar_t = rectifier(p=pt, logits=lt, w=w_t) # teacher 参照矫正

        # --- 诊断项 ---
        H_s = -(p_bar  * (p_bar.clamp_min(1e-8).log())).sum(dim=-1)  # [B]
        H_t = -(p_bar_t * (p_bar_t.clamp_min(1e-8).log())).sum(dim=-1)
        ucr = torch.relu(H_t - H_s - loss_cfg.epsilon_u).mean() if loss_cfg.enable_ucr else torch.zeros((), device=device)
        fr  = torch.relu((p_bar - p_bar_t).std(dim=0).mean() - loss_cfg.epsilon_f) if loss_cfg.enable_fr else torch.zeros((), device=device)

        # 变分 KL（轻正则）
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


# --- Reception：与 memorization 同步支持 w_mode / delta_target ---
def reception_phase(
    student, teacher, rectifier,
    labeled_rows, val_rows,
    batch_size, tokenizer, max_len,
    epochs, patience, lr,
    ema_m: float = 0.99,
    w_mode: str = "diff",
    delta_target: Optional[str] = None,
    temperature: Optional[float] = None,
    proto_source: str = "teacher"
) -> Dict[str, float]:

    if delta_target is not None:
        rectifier.target = delta_target
    if temperature is not None:
        rectifier.temperature = float(temperature)

    # 构建原型（若需要）
    protos = None
    if w_mode.startswith("proto"):
        base = teacher if (proto_source == "teacher") else student
        protos = compute_class_prototypes(base, labeled_rows, tokenizer, batch_size, max_len)

    # dataloader
    ds_tr = NliDataset(labeled_rows, tokenizer=tokenizer, max_len=max_len)
    ld_tr = build_loader(ds_tr, batch_size=batch_size, shuffle=True, tokenizer=tokenizer, max_len=max_len)

    opt = torch.optim.AdamW(student.model.parameters(), lr=lr)
    best = None; wait = 0; best_f1 = -1.0
    for ep in range(1, epochs + 1):
        student.model.train(); rectifier.eval(); teacher.model.eval()
        for batch in ld_tr:
            ps, zs, ls = _get_p_z_logits(student, batch)
            with torch.no_grad():
                pt, zt, lt = _get_p_z_logits(teacher, batch)

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

            # EMA teacher
            with torch.no_grad():
                for sp, tp in zip(student.model.parameters(), teacher.model.parameters()):
                    tp.data.mul_(ema_m).add_(sp.data * (1.0 - ema_m))

        # 简要早停逻辑（复用你现有 eval）
        # ... 这里可直接沿用你 runner 中的 eval 函数
        # 返回值保持与原先一致
        # （此处略，保持与你原有实现一致）
        # end for ep

    return {"ok": 1}  # 具体可按你原有返回替换

