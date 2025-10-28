# -*- coding: utf-8 -*-
"""
Plot Minority-F1 and Macro-F1 for BERT/DEBERTA with/without framework (v3)
Scenario: 1-1-40 (Extreme Imbalance)
- Colors by MODEL: BERT=yellow, DEBERTA=green
- Markers by FRAMEWORK: v3=hollow circle, non-v3=hollow triangle
- Legend placed OUTSIDE on the right
- Save two figures under ./figs
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
MARKER = {
    True:  "o",  # with framework (v3)
    False: "^",  # without framework
}
LINESTYLE = "-"

# 这里换成 1-1-40 的四个目录
SOURCES = [
    ("BERT_RATIO-1-1-40_v3_alvin_controlled-balanced",      "BERT",    True),
    ("BERT_RATIO-1-1-40_alvin_controlled-balanced",         "BERT",    False),
    ("DEBERTA_RATIO-1-1-40_v3_alvin_controlled-balanced",   "DEBERTA", True),
    ("DEBERTA_RATIO-1-1-40_alvin_controlled-balanced",      "DEBERTA", False),
]

# --------------- Helpers ---------------
def load_csv_from_folder(folder_path: str) -> pd.DataFrame:
    """Load the single most relevant CSV from a folder.
    If multiple CSVs exist, pick the largest (most rows)."""
    csv_files = glob.glob(os.path.join(folder_path, "*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV found in: {folder_path}")
    best_file = max(csv_files, key=lambda f: sum(1 for _ in open(f, encoding="utf-8", errors="ignore")))
    return pd.read_csv(best_file)

def tidy_df(df: pd.DataFrame, model: str, is_v3: bool) -> pd.DataFrame:
    """Standardize column types and add model/meta columns."""
    for col in ["round", "labeled_total", "minority_f1", "macro_f1"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "round" in df.columns:
        df = df.sort_values("round")
    elif "labeled_total" in df.columns:
        df = df.sort_values("labeled_total")
    df["model"] = model
    df["is_v3"] = is_v3
    return df

def collect_data() -> pd.DataFrame:
    frames = []
    for folder, model, is_v3 in SOURCES:
        path = os.path.join(ROOT, folder)
        df = load_csv_from_folder(path)
        df = tidy_df(df, model, is_v3)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)

# --------------- Plotting ---------------
def plot_metric(df: pd.DataFrame, metric: str, title: str, out_path: str):
    """Generic plotting routine for a given metric."""
    x_key = "round" if "round" in df.columns else "labeled_total"
    plt.figure(figsize=(8, 4.8))

    for (model, is_v3), g in df.groupby(["model", "is_v3"]):
        g = g.dropna(subset=[metric])
        if g.empty:
            continue
        color = COLOR.get(model, "#333333")
        marker = MARKER[is_v3]
        plt.plot(
            g[x_key].values, g[metric].values,
            LINESTYLE,
            marker=marker,
            markersize=7,
            markerfacecolor="none",
            markeredgewidth=1.6,
            color=color,
            linewidth=1.8,
            label=f"{model} ({'v3' if is_v3 else 'no-v3'})"
        )

    plt.xlabel("Round" if x_key == "round" else "Labeled Total")
    plt.ylabel(metric.replace("_", " ").title())
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

def main():
    df = collect_data()
    plot_metric(
        df=df,
        metric="minority_f1",
        title="Minority F1 vs. Round (ALVIN, Controlled-Balanced, RATIO-1-1-40)",
        out_path=os.path.join(OUTPUT_DIR, "minority_f1_1140.png")
    )
    plot_metric(
        df=df,
        metric="macro_f1",
        title="Macro F1 vs. Round (ALVIN, Controlled-Balanced, RATIO-1-1-40)",
        out_path=os.path.join(OUTPUT_DIR, "macro_f1_1140.png")
    )

if __name__ == "__main__":
    main()
