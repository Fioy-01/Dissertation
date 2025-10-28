import re
import json
from typing import Optional, List, Dict
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


LABEL_SET = {"entailment", "contradiction", "neutral"}

PROMPT_JUDGE = """You are a precise NLI judge. Given a premise and a hypothesis, answer with EXACTLY ONE WORD from this set: entailment, contradiction, neutral.
No punctuation. No explanation.

Premise: {premise}
Hypothesis: {hypothesis}
Answer:"""

PROMPT_GEN = """Generate ONE short Natural Language Inference (NLI) example as compact JSON with fields: "premise", "hypothesis", "label".
The label MUST be one of: "entailment", "contradiction", "neutral".
Keep both sentences short and everyday. Output ONLY the JSON object.
Example:
{"premise":"A man is cooking","hypothesis":"Someone is in a kitchen","label":"entailment"}

Now produce a NEW one:"""


class QwenJudge:
    def __init__(self, model_dir: str, fp16: bool = True):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True, use_fast=True, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            local_files_only=True,
            torch_dtype=(torch.float16 if fp16 and self.device.type == "cuda" else torch.float32),
            device_map="auto",
            trust_remote_code=True
        )

    @torch.no_grad()
    def _generate(self, prompt: str, max_new_tokens: int = 64, temperature: float = 0.0) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        out = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=(temperature > 0.0),
            temperature=temperature if temperature > 0 else None,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        text = self.tokenizer.decode(out[0], skip_special_tokens=True)
        # Return only the tail after the prompt
        if text.startswith(prompt):
            return text[len(prompt):].strip()
        return text.strip()

    def judge_nli_label(self, premise: str, hypothesis: str, n_retries: int = 8) -> Optional[str]:
        """
        Ask the LLM n_retries times; accept only if all answers are identical and in LABEL_SET.
        """
        answers = []
        for _ in range(n_retries):
            resp = self._generate(PROMPT_JUDGE.format(premise=premise, hypothesis=hypothesis), max_new_tokens=4, temperature=0.0)
            cand = resp.strip().lower().split()[0] if resp.strip() else ""
            cand = re.sub(r"[^a-z]", "", cand)
            if cand not in LABEL_SET:
                return None
            answers.append(cand)
        if len(set(answers)) == 1:
            return answers[0]
        return None

    def generate_short_simple(self, n_items: int, max_trials: int = 200) -> List[Dict]:
        """
        Generate up to n_items json rows with strict label check. Uses at most max_trials calls.
        """
        results = []
        trials = 0
        while len(results) < n_items and trials < max_trials:
            trials += 1
            resp = self._generate(PROMPT_GEN, max_new_tokens=128, temperature=0.0)
            # try to extract a JSON object
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
                # consistency check: ask judge 8 times
                judged = self.judge_nli_label(prem, hyp, n_retries=8)
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
