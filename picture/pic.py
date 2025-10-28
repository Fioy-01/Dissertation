# -*- coding: utf-8 -*-
"""
Active Learning Curves Plotter (Minority-F1 & Majority/Overall-F1)
- X-axis uses labeled_total (e.g., 2200, 2400, 2600, 2800).
- Colors by MODEL: BERT=red, DEBERTA=blue.
- Markers by STRATEGY: ENVA/VAR=triangle, ALVIN=square, RANDOM=circle.
- One axes per figure, Matplotlib only, no seaborn.
- Exports two PNGs under ROOT/figs/.

All plotting comments are in English (as requested).
"""

import os
import glob
import pandas as pd
import matplotlib.pyplot as plt

# ---------------- Config ----------------
# Set ROOT to the directory that contains folders like:
# BERT_RATIO-REAL_alvin_controlled-balanced, ... , DEBERTA_RATIO-REAL_random_uncontrolled
ROOT = r"./"  # <-- change to your root if needed
OUTPUT_DIR = os.path.join(ROOT, "figs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Colors by model (explicit only for models requested)
MODEL_COLOR = {
    "BERT":    "#d62728",  # red
    "DEBERTA": "#1f77b4",  # blue
}

# ---------------- Helpers ----------------
def pick_marker(strategy: str) -> str:
    """Pick marker by strategy name (case-insensitive)."""
    s = (strategy or "").lower()
    # ENVA or improved variants: names containing 'enva' or 'var'
    if "enva" in s or "var" in s:
        return "^"   # triangle
    if "alvin" in s:
        return "s"   # square
    if "random" in s:
        return "o"   # circle
    return "x"       # fallback

def infer_model_from_path(path: str) -> str:
    """Infer model name from folder/file path tokens."""
    up = path.upper()
    if "DEBERTA" in up:
        return "DEBERTA"
    if "BERT" in up:
        return "BERT"
    # try file base as fallback
    base = os.path.basename(path).upper()
    if "DEBERTA" in base:
        return "DEBERTA"
    if "BERT" in base:
        return "BERT"
    return "UNKNOWN"

def get_color(model: str):
    """Return fixed color for known models; otherwise None to let MPL choose."""
    return MODEL_COLOR.get(model, None)

def line_label(model: str, strategy: str) -> str:
    """Legend label text."""
    return f"{model} · {strategy}"

# ---------------- Load CSVs ----------------
all_rows = []
# Look for .csv recursively; adjust pattern if your files have special names
for csv_path in glob.glob(os.path.join(ROOT, "**", "*.csv"), recursive=True):
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        # Try tab-delimited fallback
        try:
            df = pd.read_csv(csv_path, sep="\t")
        except Exception:
            continue

    # Must have at least these columns
    if not {"round", "strategy"}.issubset(df.columns):
        continue

    # Attach model column if missing (from path)
    if "model" not in df.columns:
        df["model"] = infer_model_from_path(csv_path)

    df["source_file"] = csv_path
    all_rows.append(df)

if not all_rows:
    raise RuntimeError(f"No valid CSV files found under ROOT={os.path.abspath(ROOT)}")

data = pd.concat(all_rows, ignore_index=True)

# Normalize dtypes and basic cleaning
for col in ["round", "labeled_total"]:
    if col in data.columns:
        data[col] = pd.to_numeric(data[col], errors="coerce")
data["strategy"] = data["strategy"].astype(str).str.strip()
data = data.dropna(subset=["round"])  # keep valid rows

# Decide which metric to use for "majority"
HAS_MAJ = "majority_f1" in data.columns
MAJ_METRIC = "majority_f1" if HAS_MAJ else "macro_f1"
MAJ_TITLE  = "Majority-class F1 vs. Labeled Total" if HAS_MAJ else "Overall (Macro) F1 vs. Labeled Total"
MAJ_YLABEL = "Majority F1" if HAS_MAJ else "Overall (Macro) F1"

# Common x ticks: unique labeled_total sorted if present; else fallback to round
USE_LABELED_TOTAL = "labeled_total" in data.columns and data["labeled_total"].notna().any()
if USE_LABELED_TOTAL:
    xtick_vals = sorted(data["labeled_total"].dropna().unique().tolist())
else:
    xtick_vals = sorted(data["round"].dropna().unique().tolist())

# ---------------- Plotting ----------------
def plot_metric(metric: str, title: str, ylabel: str, outname: str):
    """
    Draw one figure for the given metric against labeled_total (preferred) or round.
    - Puts legend outside on the right.
    - Fixes x ticks to the observed labeled_total values (e.g. 2200, 2400, 2600, 2800).
    """
    fig, ax = plt.subplots(figsize=(10.5, 5.5))  # slightly wider to leave room for legend

    # choose x column
    xcol = "labeled_total" if USE_LABELED_TOTAL else "round"

    # group per (model, strategy)
    for (model, strategy), g in data.groupby(["model", "strategy"], dropna=False):
        if metric not in g.columns:
            continue
        gg = g.sort_values(xcol)
        if gg[metric].isna().all():
            continue

        color = get_color(model)       # enforce requested colors when known
        marker = pick_marker(strategy) # marker by strategy

        ax.plot(
            gg[xcol], gg[metric],
            marker=marker,
            linewidth=1.8,
            markersize=6,
            label=line_label(model, strategy),
            color=color  # leave None for unknown models -> MPL auto color
        )

    # Titles and labels
    ax.set_title(title)
    ax.set_xlabel("Labeled Total" if xcol == "labeled_total" else "Round")
    ax.set_ylabel(ylabel)

    # Fix x ticks to exact values and add vertical gridlines
    ax.set_xticks(xtick_vals)
    ax.grid(True, axis="both", linestyle="--", linewidth=0.6, alpha=0.7)

    # Shrink axes and place legend outside on the right-top
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.78, box.height])
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=9, frameon=True)

    # Tight + bbox_inches prevents ylabel being cut off
    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, outname)
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close(fig)

# Figure 1: minority F1
plot_metric(
    metric="minority_f1",
    title="Minority-class F1 vs. Labeled Total",
    ylabel="Minority F1",
    outname="minority_f1_vs_labeled_total.png"
)

# Figure 2: majority / overall F1
plot_metric(
    metric=MAJ_METRIC,
    title=MAJ_TITLE,
    ylabel=MAJ_YLABEL,
    outname="majority_or_overall_f1_vs_labeled_total.png"
)

print("Done.")
