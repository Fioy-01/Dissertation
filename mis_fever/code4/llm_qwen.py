# -*- coding: utf-8 -*-
"""
Qwen/通用 Chat LLM 客户端（生成 + 判别）
- 生成短/长样本（NLI）
- 判别 NLI 标签（entailment/neutral/contradiction），用于 8/8 一致性
- 默认提供 MockBackend（离线可跑），也可替换为真实 API backend

用法：
    from llm_qwen import QwenClient, MockBackend

    llm = QwenClient(backend=MockBackend(seed=42))
    one = llm.generate_short_simple_candidate()
    y = llm.judge_label(one)  # -> "entailment" / "neutral" / "contradiction"

要接入真实 LLM，只需实现一个 backend：
    class MyBackend:
        def send(self, messages: list[dict], **kwargs) -> str:
            # messages: [{"role":"system"/"user"/"assistant","content":"..."}]
            # return: 模型生成的字符串
            ...
    然后：QwenClient(backend=MyBackend(...))
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import random
import re
import json
import numpy as np

LABELS = ["entailment", "neutral", "contradiction"]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}
ID2LABEL = {i: l for l, i in LABEL2ID.items()}


# =========================
# 工具
# =========================
def _norm_label(s: str) -> Optional[str]:
    if s is None:
        return None
    t = str(s).strip().lower()
    if t in LABEL2ID:
        return t
    # 常见别名/拼写
    alias = {
        "entails": "entailment",
        "entailed": "entailment",
        "e": "entailment",
        "n": "neutral",
        "c": "contradiction",
        "contradict": "contradiction",
        "contradictory": "contradiction",
        "contradicts": "contradiction",
        "neutrality": "neutral",
        "unknown": "neutral",
    }
    return alias.get(t, None)


def _extract_label_from_text(s: str) -> Optional[str]:
    """
    从 LLM 返回文本里抽出标签（鲁棒）。
    支持 JSON 或纯文本，如：
      {"label":"entailment"} / label: contradiction / Answer: neutral
    """
    if not s:
        return None

    # 尝试 JSON
    try:
        j = json.loads(s)
        if isinstance(j, dict):
            lab = j.get("label") or j.get("answer") or j.get("prediction")
            lab = _norm_label(lab)
            if lab:
                return lab
    except Exception:
        pass

    # 正则抓取
    pats = [
        r'label\s*[:：]\s*(\w+)',
        r'answer\s*[:：]\s*(\w+)',
        r'prediction\s*[:：]\s*(\w+)',
        r'^\s*(entailment|neutral|contradiction)\s*$',
    ]
    text = s.strip()
    for p in pats:
        m = re.search(p, text, flags=re.IGNORECASE | re.MULTILINE)
        if m:
            lab = _norm_label(m.group(1))
            if lab:
                return lab

    # 最后兜底：在文本中搜索关键字
    for lab in LABELS:
        if re.search(rf"\b{lab}\b", text, flags=re.IGNORECASE):
            return lab
    return None


# =========================
# Prompt 模板
# =========================
def build_judge_prompt(premise: str, hypothesis: str) -> List[Dict[str, str]]:
    sys = (
        "You are a precise NLI judge. Decide whether the HYPOTHESIS is "
        "entailed by, contradicts, or is neutral with respect to the PREMISE. "
        "Answer with only one of: entailment / neutral / contradiction. "
        "If unsure, choose neutral."
    )
    usr = f"PREMISE: {premise}\nHYPOTHESIS: {hypothesis}\n\nAnswer with a single word label."
    return [{"role": "system", "content": sys}, {"role": "user", "content": usr}]


def build_short_simple_gen_prompt() -> List[Dict[str, str]]:
    sys = (
        "You are a data generator for Natural Language Inference (NLI). "
        "Create a short PREMISE and a short HYPOTHESIS (<= 20 words each), "
        "in everyday language, unambiguous, no named entities if possible."
    )
    usr = (
        "Return a compact JSON with fields:\n"
        '{ "premise": "...", "hypothesis": "..." }\n'
        "Do not include extra text."
    )
    return [{"role": "system", "content": sys}, {"role": "user", "content": usr}]


def build_long_simple_gen_prompt() -> List[Dict[str, str]]:
    sys = (
        "You are a data generator for NLI. "
        "Write a 3-4 sentence PREMISE as a coherent mini-scene, and a simple HYPOTHESIS "
        "that has a clear entailment/contradiction/neutral relation to the PREMISE. "
        "Avoid world knowledge; rely on the premise content. Keep language simple."
    )
    usr = (
        "Return JSON only:\n"
        '{ "premise": "SENTENCE1 SENTENCE2 SENTENCE3 (and optional SENTENCE4).",'
        '  "hypothesis": "..." }'
    )
    return [{"role": "system", "content": sys}, {"role": "user", "content": usr}]


# =========================
# Backend 抽象
# =========================
class BackendIF:
    """后端接口：只需实现 send(messages) -> str。"""

    def send(self, messages: List[Dict[str, str]], **kwargs) -> str:
        raise NotImplementedError


@dataclass
class MockBackend(BackendIF):
    """
    离线 mock：
    - 生成：模板 + 随机句片拼接
    - 判别：基于浅层词触发规则（若无明确触发，则返回 neutral）
    仅用于联调与跑通流程；请用真实 LLM 替换以获取高质量数据。
    """
    seed: int = 0

    def __post_init__(self):
        self.rng = random.Random(self.seed)
        self.np_rng = np.random.default_rng(self.seed)

        self.actions_pos = ["smiles", "agrees", "confirms", "arrives", "finishes", "starts"]
        self.actions_neg = ["denies", "rejects", "leaves", "cancels", "breaks", "stops"]
        self.entities = ["the worker", "a student", "the coach", "a child", "the neighbor", "an artist"]
        self.objects = ["the task", "the plan", "the game", "the show", "the project", "the lesson"]
        self.modifiers = ["today", "soon", "later", "already", "quickly", "slowly"]

    # 生成一句
    def _short_pair(self) -> Dict[str, str]:
        e = self.rng.choice(self.entities)
        o = self.rng.choice(self.objects)
        m = self.rng.choice(self.modifiers)
        # 随机决定关系
        rel = self.rng.choice(LABELS)
        if rel == "entailment":
            p = f"{e} {self.rng.choice(self.actions_pos)} {o} {m}."
            h = f"{e} {self.rng.choice(['does', 'will'])} {self.rng.choice(self.actions_pos)} {o}."
        elif rel == "contradiction":
            p = f"{e} {self.rng.choice(self.actions_pos)} {o} {m}."
            # 注意引号：用双引号包围 won't
            h = f"{e} {self.rng.choice(['does', 'does not', 'won\\'t'])} {self.rng.choice(self.actions_pos)} {o}."
        else:
            p = f"{e} looks at {o} {m}."
            h = f"{e} thinks about a different topic."
        return {"premise": p, "hypothesis": h, "_rel": rel}

    def _long_pair(self) -> Dict[str, str]:
        e = self.rng.choice(self.entities)
        o = self.rng.choice(self.objects)
        s1 = f"{e} arrives at the room."
        s2 = f"There is {o} prepared on a table."
        s3 = f"People wait quietly."
        s4 = f"The atmosphere feels calm."
        premise = " ".join([s1, s2, s3, s4])

        rel = self.rng.choice(LABELS)
        if rel == "entailment":
            h = f"{e} is present in the room."
        elif rel == "contradiction":
            h = f"{e} is not in the room."
        else:
            h = f"The table is made of glass."
        return {"premise": premise, "hypothesis": h, "_rel": rel}

    # 简易规则判别（仅 mock）
    def _rule_judge(self, premise: str, hypothesis: str) -> str:
        p = f"{premise} {hypothesis}".lower()
        neg_patterns = ["does not", "don't", "won't", "cannot", "not ", "never"]
        if any(x in hypothesis.lower() for x in neg_patterns) and not any(x in premise.lower() for x in neg_patterns):
            # 假设是否定而前提非否定，粗糙地倾向矛盾
            return "contradiction"
        # 若假设语义包含前提主体与动作，粗糙地倾向蕴含
        subj = any(e in p for e in [e.lower() for e in self.entities])
        act_pos = any(a in p for a in self.actions_pos)
        if subj and act_pos and ("looks at" not in hypothesis.lower()):
            return "entailment"
        return "neutral"

    def send(self, messages: List[Dict[str, str]], **kwargs) -> str:
        # 简单根据 user 内容选择行为
        user = " ".join([m["content"] for m in messages if m["role"] == "user"]).lower()
        if "return a compact json" in user and '"premise"' in user and '"hypothesis"' in user:
            ex = self._short_pair()
            return json.dumps({"premise": ex["premise"], "hypothesis": ex["hypothesis"]})
        if "return json only" in user and '"premise"' in user:
            ex = self._long_pair()
            return json.dumps({"premise": ex["premise"], "hypothesis": ex["hypothesis"]})
        if "premise:" in user and "hypothesis:" in user and "answer" in user:
            # 解析出 P/H 并规则判别
            m_p = re.search(r"premise:\s*(.*)\n", user, re.IGNORECASE)
            m_h = re.search(r"hypothesis:\s*(.*)\n", user, re.IGNORECASE)
            if m_p and m_h:
                p = m_p.group(1).strip()
                h = m_h.group(1).strip()
                lab = self._rule_judge(p, h)
                return lab
        # 兜底
        return "neutral"


# =========================
# Qwen Client
# =========================
class QwenClient:
    """
    统一的 LLM 客户端：支持生成短/长样本 & 判别标签。
    backend 只需实现 BackendIF.send(messages)->str。
    """

    def __init__(self, backend: BackendIF):
        self.backend = backend

    # --------------------
    # 判别
    # --------------------
    def judge_label(self, sample: Dict[str, Any], max_retries: int = 3) -> Optional[str]:
        premise = str(sample.get("premise", "")).strip()
        hypothesis = str(sample.get("hypothesis", "")).strip()
        if not premise or not hypothesis:
            return None
        messages = build_judge_prompt(premise, hypothesis)
        ans = None
        for _ in range(max_retries):
            try:
                raw = self.backend.send(messages)
                lab = _extract_label_from_text(raw)
                if lab in LABEL2ID:
                    ans = lab
                    break
            except Exception:
                continue
        return ans

    # --------------------
    # 生成：短
    # --------------------
    def generate_short_simple_candidate(self, max_retries: int = 3) -> Dict[str, str]:
        messages = build_short_simple_gen_prompt()
        for _ in range(max_retries):
            try:
                raw = self.backend.send(messages)
                j = json.loads(raw)
                if isinstance(j, dict) and "premise" in j and "hypothesis" in j:
                    p = str(j["premise"]).strip()
                    h = str(j["hypothesis"]).strip()
                    if p and h:
                        return {"premise": p, "hypothesis": h}
            except Exception:
                # 尝试从非 JSON 文本中提取
                m_p = re.search(r'"premise"\s*:\s*"(.*?)"', raw, flags=re.DOTALL | re.IGNORECASE)
                m_h = re.search(r'"hypothesis"\s*:\s*"(.*?)"', raw, flags=re.DOTALL | re.IGNORECASE)
                if m_p and m_h:
                    p = m_p.group(1).strip()
                    h = m_h.group(1).strip()
                    if p and h:
                        return {"premise": p, "hypothesis": h}
                continue
        # 兜底：返回一个简单样本，避免流程中断
        return {"premise": "A person looks at a clock.", "hypothesis": "Someone is checking the time."}

    def generate_short_simple(self, n_items: int, max_trials: int = 2000) -> List[Dict[str, str]]:
        out: List[Dict[str, str]] = []
        trials = 0
        while len(out) < n_items and trials < max_trials:
            trials += 1
            ex = self.generate_short_simple_candidate()
            if ex and ex.get("premise") and ex.get("hypothesis"):
                out.append(ex)
        return out[:n_items]

    # --------------------
    # 生成：长
    # --------------------
    def generate_long_simple_candidate(self, max_retries: int = 3) -> Dict[str, str]:
        messages = build_long_simple_gen_prompt()
        for _ in range(max_retries):
            try:
                raw = self.backend.send(messages)
                j = json.loads(raw)
                if isinstance(j, dict) and "premise" in j and "hypothesis" in j:
                    p = " ".join(str(j["premise"]).split())
                    h = " ".join(str(j["hypothesis"]).split())
                    if p and h:
                        return {"premise": p, "hypothesis": h}
            except Exception:
                # 尝试从非 JSON 文本中提取
                m_p = re.search(r'"premise"\s*:\s*"(.*?)"', raw, flags=re.DOTALL | re.IGNORECASE)
                m_h = re.search(r'"hypothesis"\s*:\s*"(.*?)"', raw, flags=re.DOTALL | re.IGNORECASE)
                if m_p and m_h:
                    p = " ".join(m_p.group(1).split())
                    h = " ".join(m_h.group(1).split())
                    if p and h:
                        return {"premise": p, "hypothesis": h}
                continue
        # 兜底
        return {
            "premise": "A teacher enters the quiet class. There is a book on the desk. Students wait silently.",
            "hypothesis": "A teacher is in the classroom.",
        }

    def generate_long_simple(self, n_items: int, max_trials: int = 4000) -> List[Dict[str, str]]:
        out: List[Dict[str, str]] = []
        trials = 0
        while len(out) < n_items and trials < max_trials:
            trials += 1
            ex = self.generate_long_simple_candidate()
            if ex and ex.get("premise") and ex.get("hypothesis"):
                out.append(ex)
        return out[:n_items]

    # 兼容 sampling.py 的通用入口（如果有）
    def generate(self, template: str = "short_simple", **kwargs) -> Dict[str, str]:
        if template == "short_simple":
            return self.generate_short_simple_candidate()
        if template == "long_simple":
            return self.generate_long_simple_candidate()
        # 兜底回短样本
        return self.generate_short_simple_candidate()

