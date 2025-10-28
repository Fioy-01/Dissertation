# -*- coding: utf-8 -*-
"""
Active Learning Curves Plotter (Scenario color + Ablation marker)
- X-axis uses labeled_total (e.g., 2200, 2400, 2600, 2800) instead of round
- Colors by SCENARIO: 1:1:40 -> red, real ratio -> blue
- Markers by ABLATION: FR -> square, UCR -> triangle, Original -> circle
- Two figures: (1) Minority F1 vs Labeled Total, (2) Macro F1 vs Labeled Total
All plotting comments are in English as requested.
"""

import os
import re
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ---------------- Config ----------------
ROOT = "."                  # root folder to scan, change if needed
OUTPUT_DIR = "./figs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

CSV_NAME = "round_log.csv"  # expected file name inside each run folder

# Color by scenario
SCEN_COLOR = {
    "1:1:40": "red",
    "real": "blue",
}

# Marker by ablation
ABLAT_MARKER = {
    "FR": "s",        # square
    "UCR": "^",       # triangle
    "Original": "o",  # circle
}

LINESTYLE = "-"      # single style for clarity
LINEWIDTH = 2
MARKERSIZE = 7

# ---------------- Helpers ----------------
def infer_scenario(path: str) -> str:
    """Infer scenario from folder name."""
    name = os.path.basename(path).lower()
    if "ratio-1-1-40" in name or "1_1_40" in name or "1-1-40" in name or "1140" in name:
        return "1:1:40"
    # treat everything else containing 'real' as real-ratio scenario
    if "ratio-real" in name or "_real" in name or "real_ratio" in name:
        return "real"
    # fallback: try parent path
    parent = os.path.basename(os.path.dirname(path)).lower()
    if "ratio-1-1-40" in parent or "1_1_40" in parent or "1-1-40" in parent or "1140" in parent:
        return "1:1:40"
    if "ratio-real" in parent or "_real" in parent or "real_ratio" in parent:
        return "real"
    # default to 'real' if uncertain
    return "real"

def infer_ablation(path: str) -> str:
    """Infer ablation from folder name."""
    name = os.path.basename(path).lower()
    if "abl-fr" in name or re.search(r"(_|-)fr($|_)", name):
        return "FR"
    if "abl-ucr" in name or "ucr" in name:
        return "UCR"
    return "Original"

def smart_read_round_log(csv_path: str) -> pd.DataFrame:
    """
    Read round_log with auto delimiter sniffing.
    We coerce numeric columns; unknown columns are ignored.
    """
    # sep=None lets pandas infer comma or tab
    df = pd.read_csv(csv_path, sep=None, engine="python")
    # normalize column names
    df.columns = [c.strip().lower() for c in df.columns]
    # expected numeric columns
    num_cols = [
        "round", "labeled_total", "minority_f1", "minority_recall",
        "minority_precision", "macro_f1", "accuracy",
        "macro_precision", "macro_recall"
    ]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    # keep only needed cols and drop NaNs in labeled_total
    keep = ["labeled_total", "minority_f1", "macro_f1"]
    keep = [k for k in keep if k in df.columns]
    if "labeled_total" not in keep:
        raise ValueError(f"'labeled_total' column not found in {csv_path}")
    df = df[keep].dropna(subset=["labeled_total"]).sort_values("labeled_total")
    # sometimes duplicated labeled_total; keep the last occurrence
    df = df.drop_duplicates(subset=["labeled_total"], keep="last")
    return df

def collect_runs(root: str):
    """Find all round_log.csv files under root and attach metadata."""
    records = []
    for csv_path in glob.glob(os.path.join(root, "**", CSV_NAME), recursive=True):
        run_dir = os.path.dirname(csv_path)
        scenario = infer_scenario(run_dir)
        ablation = infer_ablation(run_dir)
        try:
            df = smart_read_round_log(csv_path)
            if df.empty:
                continue
            records.append({
                "dir": run_dir,
                "scenario": scenario,
                "ablation": ablation,
                "df": df
            })
        except Exception as e:
            print(f"[WARN] Failed to read {csv_path}: {e}")
    return records

def gather_all_xticks(records):
    """Collect and sort all labeled_total values across runs for aligned x-ticks."""
    xs = []
    for rec in records:
        xs.extend(rec["df"]["labeled_total"].tolist())
    xs = sorted(set([int(x) for x in xs]))
    return xs

def plot_metric(records, metric: str, title: str, outfile: str):
    """Plot a given metric vs labeled_total for all runs."""
    plt.figure(figsize=(8, 5))
    # Build a label that combines scenario + ablation; disambiguate duplicates.
    seen_labels = set()
    for rec in records:
        df = rec["df"]
        if metric not in df.columns:
            continue
        color = SCEN_COLOR.get(rec["scenario"], "black")
        marker = ABLAT_MARKER.get(rec["ablation"], "o")
        label = f"{rec['scenario']} · {rec['ablation']}"
        # if same label appears multiple times, append a counter suffix
        lbl = label
        cnt = 2
        while lbl in seen_labels:
            lbl = f"{label} ({cnt})"
            cnt += 1
        seen_labels.add(lbl)
        # Plot line using labeled_total as x
        plt.plot(
            df["labeled_total"].values,
            df[metric].values,
            linestyle=LINESTYLE,
            linewidth=LINEWIDTH,
            marker=marker,
            markersize=MARKERSIZE,
            color=color,
            label=lbl
        )

    # x-axis ticks unified across runs (e.g., 2200, 2400, 2600, 2800)
    xticks = gather_all_xticks(records)
    if xticks:
        plt.xticks(xticks)

    plt.xlabel("Labeled Total")                  # x-axis label
    plt.ylabel(metric.replace("_", " ").title()) # y-axis label
    plt.title(title)
    plt.grid(True, alpha=0.3)
    # Legend outside on the right
    plt.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), borderaxespad=0.)
    plt.tight_layout()
    plt.savefig(outfile, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"[OK] Saved: {outfile}")

# ---------------- Run ----------------
if __name__ == "__main__":
    runs = collect_runs(ROOT)
    if not runs:
        print("[WARN] No runs found. Make sure your ROOT contains subfolders with round_log.csv")
    else:
        # Figure 1: Minority F1
        plot_metric(
            runs,
            metric="minority_f1",
            title="Minority F1 vs Labeled Total (Red=1:1:40, Blue=Real | Marker: □ FR, △ UCR, ○ Original)",
            outfile=os.path.join(OUTPUT_DIR, "minority_f1_by_scenario_ablation.png"),
        )
        # Figure 2: Macro F1
        plot_metric(
            runs,
            metric="macro_f1",
            title="Macro F1 vs Labeled Total (Red=1:1:40, Blue=Real | Marker: □ FR, △ UCR, ○ Original)",
            outfile=os.path.join(OUTPUT_DIR, "macro_f1_by_scenario_ablation.png"),
        )
