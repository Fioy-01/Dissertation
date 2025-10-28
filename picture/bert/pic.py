# -*- coding: utf-8 -*-
"""
Plot AL curves for BERT_ratio-* runs.
- X-axis uses labeled_total instead of round (e.g. 2000, 2200, 2400...)
- Colors by scenario: 1:1:40 -> red, real -> blue
- Markers by ablation: FR -> square, UCR -> triangle, Original -> circle
- Two figures: Minority F1, Macro F1
"""

import os
import re
import glob
import pandas as pd
import matplotlib.pyplot as plt

# ---------------- Config ----------------
ROOT = "."                   # root folder to scan
OUTPUT_DIR = "./figs"
CSV_NAME = "round_log.csv"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Colors by scenario
SCEN_COLOR = {
    "1:1:40": "red",
    "real": "blue",
}

# Markers by ablation
ABLAT_MARKER = {
    "FR": "s",        # square
    "UCR": "^",       # triangle
    "Original": "o",  # circle
}

LINESTYLE = "-"
LINEWIDTH = 2
MARKERSIZE = 7


# ---------------- Helpers ----------------
def infer_scenario_from_name(name: str) -> str:
    nm = name.lower()
    if ("ratio-1_1_40" in nm or "ratio-1-1-40" in nm or "1_1_40" in nm
            or "1-1-40" in nm or "1140" in nm):
        return "1:1:40"
    if "ratio-real" in nm or "_real" in nm or "real_ratio" in nm or "real-" in nm:
        return "real"
    return "real"


def infer_ablation_from_name(name: str) -> str:
    nm = name.lower()
    if "abl-fr" in nm or re.search(r"(^|[_-])fr($|[_-])", nm):
        return "FR"
    if "abl-ucr" in nm or "ucr" in nm:
        return "UCR"
    return "Original"


def smart_read_round_log(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, sep=None, engine="python")
    df.columns = [c.strip().lower() for c in df.columns]

    # ensure numeric
    for c in ["round", "labeled_total", "minority_f1", "macro_f1"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # keep relevant
    keep = [c for c in ["round", "labeled_total", "minority_f1", "macro_f1"] if c in df.columns]
    if "labeled_total" not in keep:
        raise ValueError(f"No 'labeled_total' column in {csv_path}")
    df = df[keep].dropna(subset=["labeled_total"]).sort_values("labeled_total")
    df = df.drop_duplicates(subset=["labeled_total"], keep="last")
    return df


def collect_runs(root: str):
    records = []
    for csv_path in glob.glob(os.path.join(root, "**", CSV_NAME), recursive=True):
        run_dir = os.path.dirname(csv_path)
        run_name = os.path.basename(run_dir)
        if "bert_ratio" not in run_name.lower():
            parent = os.path.basename(os.path.dirname(run_dir)).lower()
            if "bert_ratio" not in parent:
                continue
        scenario = infer_scenario_from_name(run_name)
        ablation = infer_ablation_from_name(run_name)
        try:
            df = smart_read_round_log(csv_path)
        except Exception as e:
            print(f"[WARN] Skip {csv_path}: {e}")
            continue
        if df.empty:
            continue
        records.append({
            "name": run_name,
            "scenario": scenario,
            "ablation": ablation,
            "df": df,
        })
    return records


def plot_metric(records, metric: str, title: str, outfile: str):
    plt.figure(figsize=(8, 5))
    used_labels = set()

    for rec in records:
        df = rec["df"]
        if metric not in df.columns:
            continue
        color = SCEN_COLOR.get(rec["scenario"], "black")
        marker = ABLAT_MARKER.get(rec["ablation"], "o")
        base_label = f"{rec['scenario']} · {rec['ablation']}"
        label = base_label
        k = 2
        while label in used_labels:
            label = f"{base_label} ({k})"
            k += 1
        used_labels.add(label)

        plt.plot(
            df["labeled_total"].values,
            df[metric].values,
            linestyle=LINESTYLE,
            linewidth=LINEWIDTH,
            marker=marker,
            markersize=MARKERSIZE,
            color=color,
            label=label
        )

    plt.xlabel("Labeled Total")   # <-- now use labeled_total instead of round
    plt.ylabel(metric.replace("_", " ").title())
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), borderaxespad=0.)
    plt.tight_layout()
    plt.savefig(outfile, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"[OK] Saved: {outfile}")


# ---------------- Main ----------------
if __name__ == "__main__":
    runs = collect_runs(ROOT)
    if not runs:
        print("[WARN] No runs found under ROOT. Check paths.")
    else:
        plot_metric(
            runs,
            metric="minority_f1",
            title="Minority F1 vs Labeled Total (Red=1:1:40, Blue=Real | □FR △UCR ○Original)",
            outfile=os.path.join(OUTPUT_DIR, "BERT_ratio_minority_f1.png"),
        )
        plot_metric(
            runs,
            metric="macro_f1",
            title="Macro F1 vs Labeled Total (Red=1:1:40, Blue=Real | □FR △UCR ○Original)",
            outfile=os.path.join(OUTPUT_DIR, "BERT_ratio_macro_f1.png"),
        )
