# datasets.py
import json
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

# ★ 与 sampling.py 对齐：["entailment","neutral","contradiction"]
LABEL2ID = {"entailment": 0, "neutral": 1, "contradiction": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}


def load_jsonl(path: str) -> List[Dict]:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    return data


def save_jsonl(path: str, rows: List[Dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def split_by_indices(pool: List[Dict], picked_indices: List[int]) -> Tuple[List[Dict], List[Dict]]:
    picked_set = set(picked_indices)
    picked, remain = [], []
    for i, row in enumerate(pool):
        (picked if i in picked_set else remain).append(row)
    return picked, remain


@dataclass
class NliItem:
    premise: str
    hypothesis: str
    label: Optional[int]  # ★ label 可选


class NliDataset(Dataset):
    """
    仅返回原始文本与(可选)标签；分词放到 DataLoader 的 collate_fn 里做。
    每个样本字典至少包含 'premise' 和 'hypothesis'。
    """
    def __init__(self, samples: List[Dict], tokenizer: PreTrainedTokenizerBase, max_len: int = 256):
        self.samples = samples
        self.tokenizer = tokenizer  # collate_fn 会用到
        self.max_len = max_len

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        row = self.samples[idx]
        prem = row["premise"]
        hypo = row["hypothesis"]

        out = {
            "premise": prem,
            "hypothesis": hypo,
        }

        # ★ label 可缺省：仅在存在时返回 'labels'
        if "label" in row and row["label"] is not None:
            label_name = row["label"]
            if isinstance(label_name, int):
                label_id = label_name
            else:
                label_id = LABEL2ID[label_name]
            out["labels"] = label_id

        return out

