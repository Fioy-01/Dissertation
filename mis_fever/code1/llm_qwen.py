# -*- coding: utf-8 -*-
import re
import json
from typing import Optional, List, Dict, Iterable
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

LABEL_SET = {"entailment", "contradiction", "neutral"}

PROMPT_JUDGE = """You are a precise NLI judge. Given a premise and a hypothesis, answer with EXACTLY ONE WORD from this set: entailment, contradiction, neutral.
No punctuation. No explanation.

Premise: {premise}
Hypothesis: {hypothesis}
Answer:"""

# 生成一个“短且简单”的 NLI 样本（仅给出 premise/hypothesis；不包含 label）
PROMPT_GEN_SHORT_SIMPLE = """Generate ONE short Natural Language Inference (NLI) pair as compact JSON with fields: "premise", "hypothesis".
- Keep both sentences short, everyday, single sentence each.
- Do NOT include any label field.
- Output ONLY the JSON object.

Example:
{"premise":"A man is cooking","hypothesis":"Someone is in a kitchen"}

Now produce a NEW one:"""

# 你原来带 label 的短样本（保留，向后兼容）
PROMPT_GEN_SHORT_WITH_LABEL = """Generate ONE short Natural Language Inference (NLI) example as compact JSON with fields: "premise", "hypothesis", "label".
The label MUST be one of: "entailment", "contradiction", "neutral".
Keep both sentences short and everyday. Output ONLY the JSON object.
Example:
{"premise":"A man is cooking","hypothesis":"Someone is in a kitchen","label":"entailment"}

Now produce a NEW one:"""

# Long & Simple：四句前提 + 简单关系假设（不要求跨句推理）
PROMPT_GEN_LONG_SIMPLE = """Create ONE NLI example where the premise has EXACTLY FOUR short sentences, and the hypothesis is ONE short sentence.
Return compact JSON with fields: "premise", "hypothesis".
Constraints:
- Premise: four sentences, everyday topics, concise.
- Hypothesis: directly supported or related in a simple way (no multi-hop reasoning).
- Do NOT include a label field.
- Output ONLY the JSON object.

Example:
{"premise":"The cafe opens at 8am. A barista is setting up cups. A few customers wait outside. The weather is cold.","hypothesis":"The cafe is not open yet"}

Now produce a NEW one:"""

# 难度分提示：只返回 1..10 的整数
PROMPT_DIFFICULTY = """You are rating the difficulty of an NLI example for a fine-tuned classifier.
Score 1..10 (10 = hardest), output ONLY the integer, no text.

Premise: {premise}
Hypothesis: {hypothesis}
Gold label: {label}
Score:"""


# ---------------------------
# 生成参数白名单过滤器（类外）
# ---------------------------
def _sanitize_gen_kwargs_for_hf(model, do_sample: bool, **kwargs):
    """
    仅当“生成模型 + 启用采样”时保留采样相关键；
    其余情况（分类模型/未采样）全部剔除，避免
    'The following generation flags are not valid...' 提示。
    """
    if not do_sample:
        return {}

    gen_cfg = getattr(model, "generation_config", None)
    if gen_cfg is None or not hasattr(model, "generate"):
        return {}

    # generation_config 可识别的键 + 常见控制键
    try:
        supported = set(gen_cfg.to_dict().keys())
    except Exception:
        supported = set()
    supported |= {
        "do_sample", "temperature", "top_p", "top_k",
        "max_new_tokens", "min_new_tokens",
        "num_beams", "num_return_sequences",
        "repetition_penalty", "length_penalty",
        "no_repeat_ngram_size", "early_stopping",
        "eos_token_id", "pad_token_id", "use_cache"
    }
    clean = {k: v for k, v in kwargs.items() if (v is not None and k in supported)}
    clean["do_sample"] = True
    return clean


class QwenJudge:
    def __init__(self, model_dir: str, fp16: bool = True):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_dir, local_files_only=True, use_fast=True, trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            local_files_only=True,
            torch_dtype=(torch.float16 if fp16 and self.device.type == "cuda" else torch.float32),
            device_map="auto",
            trust_remote_code=True
        )

    # ---------------------------
    # 基础生成
    # ---------------------------
    @torch.no_grad()
    def _generate(
        self,
        prompt: str,
        max_new_tokens: int = 96,
        temperature: float = 0.0,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
    ) -> str:
        # 兜底 pad/eos，减少无关提示
        eos_id = self.tokenizer.eos_token_id
        pad_id = getattr(self.tokenizer, "pad_token_id", None) or eos_id

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        # 只有在真的需要采样时才传采样键
        do_sample = False
        if temperature is not None and float(temperature) > 0.0:
            do_sample = True
        if top_p is not None and float(top_p) < 1.0:
            do_sample = True
        if top_k is not None and int(top_k) > 0:
            do_sample = True

        base_kwargs = dict(
            max_new_tokens=int(max_new_tokens),
            eos_token_id=eos_id,
            pad_token_id=pad_id,
            use_cache=True,
            do_sample=do_sample,
        )

        sample_kwargs = _sanitize_gen_kwargs_for_hf(
            self.model,
            do_sample=do_sample,
            temperature=float(temperature) if temperature is not None else None,
            top_p=float(top_p) if top_p is not None else None,
            top_k=int(top_k) if top_k is not None else None,
            max_new_tokens=int(max_new_tokens),
            eos_token_id=eos_id,
            pad_token_id=pad_id,
        )

        out = self.model.generate(**inputs, **base_kwargs, **sample_kwargs)
        text = self.tokenizer.decode(out[0], skip_special_tokens=True)
        # 去掉前缀 prompt
        if text.startswith(prompt):
            text = text[len(prompt):]
        return text.strip()

    # ---------------------------
    # 判别（供 sampling 的 _judge_label_many 调用）
    # ---------------------------
    def judge_label(self, sample: Dict) -> Optional[str]:
        """单次判别，返回 e/n/c；sampling 会自行多次调用做 8/8 一致性。"""
        prem = sample.get("premise", "").strip()
        hyp = sample.get("hypothesis", "").strip()
        if not prem or not hyp:
            return None
        resp = self._generate(
            PROMPT_JUDGE.format(premise=prem, hypothesis=hyp),
            max_new_tokens=4,
            temperature=0.0
        )
        tok = resp.strip().lower().split()[0] if resp.strip() else ""
        tok = re.sub(r"[^a-z]", "", tok)
        return tok if tok in LABEL_SET else None

    # 带重试一致性（向后兼容；sampling 不直接用）
    def judge_nli_label(self, premise: str, hypothesis: str, n_retries: int = 4) -> Optional[str]:
        answers = []
        for _ in range(n_retries):
            resp = self._generate(
                PROMPT_JUDGE.format(premise=premise, hypothesis=hypothesis),
                max_new_tokens=4,
                temperature=0.0
            )
            cand = resp.strip().lower().split()[0] if resp.strip() else ""
            cand = re.sub(r"[^a-z]", "", cand)
            if cand not in LABEL_SET:
                return None
            answers.append(cand)
        return answers[0] if len(set(answers)) == 1 else None

    # ---------------------------
    # 生成：Short & Simple
    # ---------------------------
    def generate_short_simple_candidate(self) -> Dict:
        """只生成 {premise,hypothesis}，不含 label。"""
        resp = self._generate(
            PROMPT_GEN_SHORT_SIMPLE,
            max_new_tokens=128,
            temperature=0.2, top_p=0.9, top_k=50  # 需要多样性时可保留；过滤器会在非采样时剔除
        )
        m = re.search(r"\{.*\}", resp, re.DOTALL)
        if not m:
            return {}
        try:
            obj = json.loads(m.group(0))
            prem = str(obj.get("premise", "")).strip()
            hyp  = str(obj.get("hypothesis", "")).strip()
            if prem and hyp:
                return {"premise": prem, "hypothesis": hyp}
        except Exception:
            pass
        return {}

    # 兼容你原先的批量接口（返回含 label 的样本列表）
    def generate_short_simple(self, n_items: int, max_trials: int = 200) -> List[Dict]:
        results = []
        trials = 0
        while len(results) < n_items and trials < max_trials:
            trials += 1
            resp = self._generate(
                PROMPT_GEN_SHORT_WITH_LABEL,
                max_new_tokens=128,
                temperature=0.2, top_p=0.9, top_k=50
            )
            m = re.search(r"\{.*\}", resp, re.DOTALL)
            if not m:
                continue
            try:
                obj = json.loads(m.group(0))
                prem = obj.get("premise", "").strip()
                hyp = obj.get("hypothesis", "").strip()
                lab = obj.get("label", "").strip().lower()
                if not prem or not hyp or lab not in LABEL_SET:
                    continue
                judged = self.judge_nli_label(prem, hyp, n_retries=4)
                if judged is None or judged != lab:
                    continue
                results.append({
                    "premise": prem,
                    "hypothesis": hyp,
                    "label": lab,
                    "source": "short_simple_gen"
                })
            except Exception:
                continue
        return results

    # ---------------------------
    # 生成：Long & Simple（四句前提 + 简单关系）
    # ---------------------------
    def generate_long_simple_candidate(self) -> Dict:
        """生成 {premise,hypothesis}；premise 期望为四句。"""
        resp = self._generate(
            PROMPT_GEN_LONG_SIMPLE,
            max_new_tokens=192,
            temperature=0.4, top_p=0.9, top_k=50
        )
        m = re.search(r"\{.*\}", resp, re.DOTALL)
        if not m:
            return {}
        try:
            obj = json.loads(m.group(0))
            prem = str(obj.get("premise", "")).strip()
            hyp  = str(obj.get("hypothesis", "")).strip()
            # 简单校验：句号/分句数量 >= 3 视作四句左右
            if prem and hyp and len(re.split(r"[.!?]+", prem)) >= 4:
                return {"premise": prem, "hypothesis": hyp}
        except Exception:
            pass
        return {}

    # 可选：批量封装，便于 fallback 调用
    def generate_long_simple(self, n_items: int, max_trials: int = 400) -> List[Dict]:
        out = []
        trials = 0
        while len(out) < n_items and trials < max_trials:
            trials += 1
            cand = self.generate_long_simple_candidate()
            if cand:
                out.append(cand)
        return out

    # ---------------------------
    # 难度分数（1..10）
    # ---------------------------
    def score_difficulty(self, sample: Dict, fewshot_examples: Optional[List[Dict]] = None) -> int:
        prem = sample.get("premise", "").strip()
        hyp  = sample.get("hypothesis", "").strip()
        lab  = sample.get("label", "").strip().lower()
        if not prem or not hyp or lab not in LABEL_SET:
            return 5  # 保守中间值
        prompt = PROMPT_DIFFICULTY.format(premise=prem, hypothesis=hyp, label=lab)
        if fewshot_examples:
            # 轻量 few-shot：把 1~2 个例子串起来（避免 prompt 过长）
            few = fewshot_examples[:2]
            buf = []
            for ex in few:
                buf.append(f"Example — Premise: {ex['premise']}\nHypothesis: {ex['hypothesis']}\nGold label: {ex['label']}\nScore: {ex.get('score','7')}")
            prompt = "\n".join(buf) + "\n\n" + prompt
        resp = self._generate(prompt, max_new_tokens=4, temperature=0.0)
        m = re.search(r"\d+", resp)
        if not m:
            return 5
        try:
            s = int(m.group(0))
            return int(min(10, max(1, s)))
        except Exception:
            return 5

    def score_difficulty_batch(self, samples: Iterable[Dict], fewshot_examples: Optional[List[Dict]] = None) -> List[int]:
        return [self.score_difficulty(s, fewshot_examples=fewshot_examples) for s in samples]

