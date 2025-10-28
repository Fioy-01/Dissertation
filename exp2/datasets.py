import json
from typing import List, Dict, Tuple, Optional
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
    A simple PyTorch dataset for NLI.
    """
    def __init__(self, samples: List[Dict], tokenizer: PreTrainedTokenizerBase, max_len: int = 256):
        self.samples = samples
        self.tokenizer = tokenizer
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
        enc = self.tokenizer(
            prem,
            hypo,
            padding=False,
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
        )
        item = {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": label_id
        }
        return item
