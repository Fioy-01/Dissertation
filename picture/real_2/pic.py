# -*- coding: utf-8 -*-
"""
Active Learning Curves Plotter (Minority-F1 & Majority/Overall-F1)
- X-axis uses labeled_total (e.g., 2200, 2400, 2600, 2800).
- Colors by model: BERT=red, DEBERTA=blue (as requested).
- Markers by strategy: ENVA/VAR=triangle, ALVIN=square, RANDOM=circle.
- One Axes per figure, Matplotlib only (no seaborn).
- Exports two PNGs under ROOT/figs/.

All plotting comments are in English (as requested).
"""

import os
import glob
import pandas as pd
import matplotlib.pyplot as plt

# ---------------- Config ----------------
# Set ROOT to the directory that contains your experiment subfolders.
ROOT = r"./"  # <--- change to your root folder if needed
OUTPUT_DIR = os.path.join(ROOT, "figs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Explicit colors for requested models
MODEL_COLOR = {
    "BERT":    "#d62728",  # red
    "DEBERTA": "#1f77b4",  # blue
}

# Accept common aliases for labeled_total seen in your logs (e.g., 'labeled_tc')
LABEL_TOTAL_ALIASES = ["labeled_total", "labeled_tc", "labeled_t", "labeled", "n_labeled"]

# ---------------- Helpers ----------------
def normalize_labeled_total(df: pd.DataFrame) -> pd.DataFrame:
    """Unify any alias column to 'labeled_total' and cast to numeric."""
    if "labeled_total" not in df.columns:
        for c in LABEL_TOTAL_ALIASES:
            if c in df.columns:
                df = df.rename(columns={c: "labeled_total"})
                break
    if "labeled_total" in df.columns:
        df["labeled_total"] = pd.to_numeric(df["labeled_total"], errors="coerce")
    return df

def pick_marker(strategy: str) -> str:
    """Pick marker by strategy name (case-insensitive)."""
    s = (strategy or "").lower()
    if "enva" in s or "var" in s:
        return "^"   # triangle for ENVA / improved variants
    if "alvin" in s:
        return "s"   # square for ALVIN
    if "random" in s:
        return "o"   # circle for Random
    return "x"       # fallback

def infer_model_from_path(path: str) -> str:
    """Infer model name from folder/file path tokens."""
    up = path.upper()
    if "DEBERTA" in up:
        return "DEBERTA"
    if "BERT" in up:
        return "BERT"
    base = os.path.basename(path).upper()
    if "DEBERTA" in base:
        return "DEBERTA"
    if "BERT" in base:
        return "BERT"
    return "UNKNOWN"

def get_color(model: str):
    """Return fixed color for known models; otherwise None to let Matplotlib choose."""
    return MODEL_COLOR.get(model, None)

def line_label(model: str, strategy: str) -> str:
    """Legend label text."""
    return f"{model} · {strategy}"

# ---------------- Load CSVs ----------------
all_rows = []
for csv_path in glob.glob(os.path.join(ROOT, "**", "*.csv"), recursive=True):
    # Try to read CSV (comma first, then tab fallback)
    df = None
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        try:
            df = pd.read_csv(csv_path, sep="\t")
        except Exception:
            continue
    if df is None:
        continue

    # Normalize labeled_total aliases
    df = normalize_labeled_total(df)

    # We need at least strategy + (round or labeled_total)
    if "strategy" not in df.columns:
        continue
    has_round = "round" in df.columns
    has_labeled = "labeled_total" in df.columns and df["labeled_total"].notna().any()
    if not (has_round or has_labeled):
        continue

    # Add model if missing (infer from path)
    if "model" not in df.columns:
        df["model"] = infer_model_from_path(csv_path)

    # Cast numerics
    if has_round:
        df["round"] = pd.to_numeric(df["round"], errors="coerce")

    df["source_file"] = csv_path
    all_rows.append(df)

if not all_rows:
    raise RuntimeError(f"No valid CSV files found under ROOT={os.path.abspath(ROOT)}")

data = pd.concat(all_rows, ignore_index=True)

# Global x-axis preference
USE_LABELED_TOTAL = "labeled_total" in data.columns and data["labeled_total"].notna().any()
xtick_vals = sorted(data["labeled_total"].dropna().unique().tolist()) if USE_LABELED_TOTAL \
             else sorted(data["round"].dropna().unique().tolist())

# Decide metric for "majority": prefer majority_f1; fallback to macro_f1
HAS_MAJ = "majority_f1" in data.columns
MAJ_METRIC = "majority_f1" if HAS_MAJ else "macro_f1"
MAJ_TITLE  = "Majority-class F1 vs. Labeled Total" if HAS_MAJ else "Overall (Macro) F1 vs. Labeled Total"
MAJ_YLABEL = "Majority F1" if HAS_MAJ else "Overall (Macro) F1"

# ---------------- Plotting ----------------
def plot_metric(metric: str, title: str, ylabel: str, outname: str):
    """Draw one figure for the given metric (y) against labeled_total (preferred) or round (x)."""
    fig, ax = plt.subplots(figsize=(10.5, 5.5))  # wider to accommodate legend

    xcol = "labeled_total" if USE_LABELED_TOTAL else "round"

    # Group by (model, strategy) and draw lines
    for (model, strategy), g in data.groupby(["model", "strategy"], dropna=False):
        if metric not in g.columns:
            continue
        gg = g.sort_values(xcol)
        if gg[metric].isna().all() or gg[xcol].isna().all():
            continue

        color = get_color(model)      # enforce requested colors for BERT/DEBERTA
        marker = pick_marker(strategy)

        ax.plot(
            gg[xcol], gg[metric],
            marker=marker,
            linewidth=1.8,
            markersize=6,
            label=line_label(model, strategy),
            color=color  # leave None for unknown -> Matplotlib auto color
        )

    # Axes titles/labels
    ax.set_title(title)
    ax.set_xlabel("Labeled Total" if xcol == "labeled_total" else "Round")
    ax.set_ylabel(ylabel)

    # Fix x ticks to exact values (e.g., 2200, 2400, 2600, 2800)
    if len(xtick_vals) > 0:
        ax.set_xticks(xtick_vals)

    # Grid and legend outside
    ax.grid(True, axis="both", linestyle="--", linewidth=0.6, alpha=0.7)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.78, box.height])  # make room for legend
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=9, frameon=True)

    # Save (bbox_inches='tight' prevents ylabel cropping)
    out_path = os.path.join(OUTPUT_DIR, outname)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close(fig)

# Figure 1: minority F1
plot_metric(
    metric="minority_f1",
    title="Minority-class F1 vs. Labeled Total" if USE_LABELED_TOTAL else "Minority-class F1 vs. Round",
    ylabel="Minority F1",
    outname="minority_f1.png"
)

# Figure 2: majority or overall F1
plot_metric(
    metric=MAJ_METRIC,
    title=MAJ_TITLE if USE_LABELED_TOTAL else "Overall (Macro) F1 vs. Round" if MAJ_METRIC == "macro_f1" else "Majority-class F1 vs. Round",
    ylabel=MAJ_YLABEL,
    outname="majority_or_overall_f1.png"
)

print("Done.")
