#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Class ratio checker for JSONL datasets.

- Reads a JSONL file where each line is a JSON object.
- Extracts the label and normalizes to one of:
  {"entailment", "neutral", "contradiction"}.
- Prints counts and percentages.

Supported label keys (in order): --label-key argument, "label",
"gold_label", "verdict".
FEVER-style labels are mapped automatically:
  "SUPPORTS" -> "entailment"
  "REFUTES"  -> "contradiction"
  "NOT ENOUGH INFO" -> "neutral"
"""

import argparse
import json
from collections import Counter
from typing import Optional

FEVER_MAP = {
    "supports": "entailment",
    "refutes": "contradiction",
    "not enough info": "neutral",
    "nei": "neutral",
}

VALID = {"entailment", "neutral", "contradiction"}

def normalize_label(raw: Optional[str]) -> Optional[str]:
    """Normalize raw label text to the target set or return None if unknown."""
    if raw is None:
        return None
    s = str(raw).strip()
    if not s:
        return None

    low = s.lower()

    # direct match
    if low in VALID:
        return low

    # map from FEVER terms
    if low in FEVER_MAP:
        return FEVER_MAP[low]

    # tolerate some common variants
    alias = {
        "entailed": "entailment",
        "entails": "entailment",
        "entail": "entailment",
        "contradict": "contradiction",
        "contradicted": "contradiction",
        "contra": "contradiction",
        "neutrality": "neutral",
    }
    if low in alias:
        return alias[low]

    return None

def get_label(obj: dict, prefer_key: Optional[str] = None) -> Optional[str]:
    """Extract label from a JSON obj using prefer_key or common fallbacks."""
    keys = [prefer_key] if prefer_key else []
    keys += ["label", "gold_label", "verdict"]
    for k in keys:
        if not k:
            continue
        if k in obj:
            lab = normalize_label(obj[k])
            if lab is not None:
                return lab
    # also scan shallow string values as a last resort
    for v in obj.values():
        lab = normalize_label(v if isinstance(v, str) else None)
        if lab is not None:
            return lab
    return None

def main():
    ap = argparse.ArgumentParser(description="Check class ratios in a JSONL file.")
    ap.add_argument("path", nargs="?", default="init_labeled.jsonl",
                    help="Path to JSONL file (default: init_labeled.jsonl)")
    ap.add_argument("--label-key", default=None,
                    help="Explicit key name that holds the label (optional)")
    ap.add_argument("--skip-invalid", action="store_true",
                    help="Skip lines with unknown/invalid labels (default: count as 'unknown')")
    ap.add_argument("--out-csv", default=None,
                    help="Optional path to write a one-line CSV summary")
    args = ap.parse_args()

    counts = Counter()
    total_lines = 0
    bad_lines = 0

    with open(args.path, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, start=1):
            total_lines += 1
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                bad_lines += 1
                counts["__parse_error__"] += 1
                continue

            lab = get_label(obj, args.label_key)
            if lab is None:
                bad_lines += 1
                if args.skip_invalid:
                    continue
                counts["unknown"] += 1
            else:
                counts[lab] += 1

    known_total = sum(counts.get(k, 0) for k in VALID)
    denom = known_total if args.skip_invalid else (known_total + counts.get("unknown", 0))

    def pct(n: int) -> float:
        return (100.0 * n / denom) if denom > 0 else 0.0

    print("=== Class Ratio Report ===")
    print(f"File: {args.path}")
    print(f"Total lines: {total_lines}")
    if counts.get("__parse_error__"):
        print(f"JSON parse errors: {counts['__parse_error__']}")
    if not args.skip_invalid and counts.get("unknown"):
        print(f"Unknown labels: {counts['unknown']} (included in denominator)")
    elif args.skip_invalid and (counts.get("unknown") or bad_lines):
        print(f"Skipped invalid/unknown labels: {counts.get('unknown', 0)} | bad_lines={bad_lines}")

    for k in ["entailment", "neutral", "contradiction"]:
        n = counts.get(k, 0)
        print(f"{k:15s}: {n:6d}  ({pct(n):6.2f}%)")

    print("--------------------------")
    print(f"Known total: {known_total}")
    if not args.skip_invalid:
        print(f"Denominator (known + unknown): {denom}")

    # Optional CSV
    if args.out_csv:
        import csv
        with open(args.out_csv, "w", newline="", encoding="utf-8") as cf:
            writer = csv.DictWriter(cf, fieldnames=[
                "file", "total_lines", "entailment", "neutral", "contradiction",
                "unknown", "known_total", "denominator"
            ])
            writer.writeheader()
            writer.writerow({
                "file": args.path,
                "total_lines": total_lines,
                "entailment": counts.get("entailment", 0),
                "neutral": counts.get("neutral", 0),
                "contradiction": counts.get("contradiction", 0),
                "unknown": counts.get("unknown", 0),
                "known_total": known_total,
                "denominator": denom,
            })
        print(f"CSV summary written to: {args.out_csv}")

if __name__ == "__main__":
    main()
"""
"""
