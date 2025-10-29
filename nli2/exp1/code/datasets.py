import json
from typing import List, Dict, Tuple
from dataclasses import dataclass
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

LABEL2ID = {"entailment": 0, "contradiction": 1, "neutral": 2}
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
    picked = []
    remain = []
    for i, row in enumerate(pool):
        (picked if i in picked_set else remain).append(row)
    return picked, remain


@dataclass
class NliItem:
    premise: str
    hypothesis: str
    label: int


class NliDataset(Dataset):
    """
    仅返回原始文本与标签；真正的分词在 DataLoader 的 collate_fn 里做（批量更高效）。
    """
    def __init__(self, samples: List[Dict], tokenizer: PreTrainedTokenizerBase, max_len: int = 256):
        self.samples = samples
        self.tokenizer = tokenizer  # 兼容占位，不在 __getitem__ 里使用
        self.max_len = max_len

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        row = self.samples[idx]
        prem = row["premise"]
        hypo = row["hypothesis"]
        label_name = row["label"]
        if isinstance(label_name, int):
            label_id = label_name
        else:
            label_id = LABEL2ID[label_name]
        return {
            "premise": prem,
            "hypothesis": hypo,
            "labels": label_id
        }

