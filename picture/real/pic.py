# -*- coding: utf-8 -*-
"""
Plot Minority-F1 and Macro-F1 for BERT/DEBERTA with/without framework (v3)
- Colors by MODEL: BERT=yellow, DEBERTA=green
- Markers by FRAMEWORK: v3=hollow circle, non-v3=hollow triangle
- Legend placed OUTSIDE on the right
- Save two figures under ./figs

Folder expectations (siblings of this script):
  - BERT_RATIO-REAL_v3_alvin_controlled-balanced
  - BERT_RATIO-REAL_alvin_controlled-balanced
  - DEBERTA_RATIO-REAL_v3_alvin_controlled-balanced
  - DEBERTA_RATIO-REAL_alvin_controlled-balanced

Each folder should contain a CSV (e.g., round_log.csv) with columns:
  timestamp,strategy,mode,round,labeled_total,B,minority_class,
  minority_f1,minority_recall,minority_precision,macro_f1,accuracy,
  macro_precision,macro_recall,pi_e
"""
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt

# --------------- Config ---------------
ROOT = "."  # root directory holding the four folders
OUTPUT_DIR = "./figs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Model colors (explicit)
COLOR = {
    "BERT":    "#f1c40f",  # yellow
    "DEBERTA": "#2ecc71",  # green
}

# Marker style mapping
# v3 (with framework) => hollow circle 'o'; non-v3 (without framework) => hollow triangle '^'
MARKER = {
    True:  "o",  # with framework (v3)
    False: "^",  # without framework
}

# Line style for better readability
LINESTYLE = "-"

# Mapping of folders to (model, is_v3)
SOURCES = [
    ("BERT_RATIO-REAL_v3_alvin_controlled-balanced",      "BERT",    True),
    ("BERT_RATIO-REAL_alvin_controlled-balanced",         "BERT",    False),
    ("DEBERTA_RATIO-REAL_v3_alvin_controlled-balanced",   "DEBERTA", True),
    ("DEBERTA_RATIO-REAL_alvin_controlled-balanced",      "DEBERTA", False),
]

# --------------- Helpers ---------------
def load_csv_from_folder(folder_path: str) -> pd.DataFrame:
    """Load the single most relevant CSV from a folder.
    If multiple CSVs exist, pick the largest (most rows)."""
    csv_files = glob.glob(os.path.join(folder_path, "*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV found in: {folder_path}")
    # pick the CSV with the most rows
    best_file = None
    best_rows = -1
    for f in csv_files:
        try:
            df = pd.read_csv(f)
            if len(df) > best_rows:
                best_rows = len(df)
                best_file = f
        except Exception:
            continue
    if best_file is None:
        raise RuntimeError(f"Failed to read any CSV in: {folder_path}")
    df = pd.read_csv(best_file)
    return df

def tidy_df(df: pd.DataFrame, model: str, is_v3: bool) -> pd.DataFrame:
    """Standardize column types and add model/meta columns."""
    # Ensure numeric typing for key columns
    for col in ["round", "labeled_total", "minority_f1", "macro_f1"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    # Sort by round if present; otherwise by labeled_total
    if "round" in df.columns:
        df = df.sort_values("round")
    elif "labeled_total" in df.columns:
        df = df.sort_values("labeled_total")
    df["model"] = model
    df["is_v3"] = is_v3
    return df

def collect_data() -> pd.DataFrame:
    """Load and concatenate all sources."""
    frames = []
    for folder, model, is_v3 in SOURCES:
        path = os.path.join(ROOT, folder)
        df = load_csv_from_folder(path)
        df = tidy_df(df, model, is_v3)
        frames.append(df)
    all_df = pd.concat(frames, ignore_index=True)
    return all_df

# --------------- Plotting ---------------
def plot_metric(df: pd.DataFrame, metric: str, title: str, out_path: str):
    """Generic plotting routine for a given metric.
    - x-axis: round if available; otherwise labeled_total
    - One line per (model, is_v3)
    """
    # Decide x-axis
    x_key = "round" if "round" in df.columns else "labeled_total"

    plt.figure(figsize=(8, 4.8))  # single, clean figure

    # Group by model and framework flag
    for (model, is_v3), g in df.groupby(["model", "is_v3"]):
        g = g.dropna(subset=[metric])
        if g.empty:
            continue

        # Select style
        color = COLOR.get(model, "#333333")
        marker = MARKER[is_v3]

        # Plot with hollow markers (facecolors='none')
        plt.plot(
            g[x_key].values, g[metric].values,
            LINESTYLE,
            marker=marker,
            markersize=7,
            markerfacecolor="none",   # hollow markers
            markeredgewidth=1.6,
            color=color,
            linewidth=1.8,
            label=f"{model} ({'v3' if is_v3 else 'no-v3'})"
        )

    # Labels and title
    plt.xlabel("Round" if x_key == "round" else "Labeled Total")  # x-axis label
    plt.ylabel(metric.replace("_", " ").title())                 # y-axis label
    plt.title(title)

    # Grid for readability
    plt.grid(True, linestyle="--", alpha=0.4)

    # Legend outside on the right
    plt.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

def main():
    df = collect_data()

    # Figure 1: minority_f1
    plot_metric(
        df=df,
        metric="minority_f1",
        title="Minority F1 vs. Round (ALVIN, Controlled-Balanced, RATIO-REAL)",
        out_path=os.path.join(OUTPUT_DIR, "minority_f1.png")
    )

    # Figure 2: macro_f1
    plot_metric(
        df=df,
        metric="macro_f1",
        title="Macro F1 vs. Round (ALVIN, Controlled-Balanced, RATIO-REAL)",
        out_path=os.path.join(OUTPUT_DIR, "macro_f1.png")
    )

if __name__ == "__main__":
    main()
