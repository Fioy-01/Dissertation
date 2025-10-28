# -*- coding: utf-8 -*-
"""
Active Learning Curves Plotter (per scenario => four figures total)
- Recursively scans a root directory for CSV logs (default: **/round_log.csv).
- Splits plots by SCENARIO (e.g., RATIO-REAL vs RATIO-1-1-40).
- Produces FOUR figures when both scenarios exist:
    REAL:     (1) Minority F1 vs round, (2) Macro F1 vs round
    1-1-40:   (3) Minority F1 vs round, (4) Macro F1 vs round
- Color by MODE:
    controlled-balanced = red
    uncontrolled        = blue
- Marker by STRATEGY:
    random      = triangle (^)
    uncertainty = square (s)
    alvin       = circle (o)
- Uses matplotlib only; one chart per figure (no subplots).

Usage examples:
    python al_plotter_per_scenario.py --root . --pattern round_log.csv --out ./figs --title-prefix BERT
    # Only include files whose path contains these substrings (flag can repeat)
    python al_plotter_per_scenario.py --root . --include-substr RATIO-REAL --include-substr BERT

CSV required columns (case-sensitive):
    timestamp,strategy,mode,round,labeled_total,B,minority_class,
    minority_f1,minority_recall,minority_precision,macro_f1,accuracy,macro_precision,macro_recall
"""

import os
import re
import glob
import argparse
from typing import Tuple, List

import pandas as pd
import matplotlib.pyplot as plt


# ---------------- Palettes requested by the user ----------------
# Color by MODE
MODE_COLOR = {
    "controlled-balanced": "#d62728",  # red
    "uncontrolled": "#1f77b4",         # blue
}

# Marker by STRATEGY
STRATEGY_MARKER = {
    "random": "^",        # triangle
    "entropy": "s",   # square
    "alvin": "o",         # circle
}


# ---------------- Helpers ----------------
def infer_model_and_scenario(folder_name: str) -> Tuple[str, str]:
    """
    Infer (model, scenario) from a folder name like:
        BERT_RATIO-1-1-40_alvin_controlled-balanced
        BERT_RATIO-REAL_random_uncontrolled
    Heuristic:
        model   = text before "_RATIO"
        scenario= the token immediately after "RATIO-"
    """
    model = None
    scenario = None
    m = re.match(r"([^_]+)_RATIO-(.+)", folder_name)
    if m:
        model = m.group(1)
        rest = m.group(2)
        scenario = rest.split("_")[0]  # e.g., "REAL" or "1-1-40"
    return (model or "MODEL", scenario or "SCENARIO")


def load_all_csvs(root: str, pattern: str) -> pd.DataFrame:
    """Recursively load all CSVs matching pattern in subdirs and attach meta columns."""
    paths = glob.glob(os.path.join(root, "**", pattern), recursive=True)
    frames = []
    for p in paths:
        try:
            df = pd.read_csv(p)
            folder = os.path.basename(os.path.dirname(p))
            model, scenario = infer_model_and_scenario(folder)
            df["__model"] = model
            df["__scenario"] = scenario
            df["__src"] = p
            frames.append(df)
        except Exception as e:
            print(f"[WARN] Skipping {p}: {e}")
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure required columns/dtypes exist and clean rows."""
    required = [
        "strategy", "mode", "round",
        "minority_f1", "macro_f1"
    ]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    # Cast numeric columns
    df["round"] = pd.to_numeric(df["round"], errors="coerce")
    df["minority_f1"] = pd.to_numeric(df["minority_f1"], errors="coerce")
    df["macro_f1"] = pd.to_numeric(df["macro_f1"], errors="coerce")

    # Drop invalid rows and sort
    df = df.dropna(subset=["round", "minority_f1", "macro_f1"])
    df = df.sort_values(["__model", "__scenario", "strategy", "mode", "round"])
    return df


def series_label(row_or_group: pd.Series) -> str:
    """Legend label for a series."""
    model = row_or_group["__model"]
    scenario = row_or_group["__scenario"]
    strategy = row_or_group["strategy"]
    mode = row_or_group["mode"]
    return f"{model} | {scenario} | {strategy} | {mode}"


def pick_color(mode: str) -> str:
    """Pick color by mode; fallback gray."""
    return MODE_COLOR.get(mode, "#7f7f7f")


def pick_marker(strategy: str) -> str:
    """Pick marker by strategy; fallback circle."""
    return STRATEGY_MARKER.get(strategy, "o")


def plot_metric(df: pd.DataFrame, metric: str, title: str, outfile: str) -> None:
    """Plot a single metric vs round with lines + markers. One figure only."""
    # Create a figure (no subplots)
    fig, ax = plt.subplots(figsize=(8, 5))

    # Group and draw
    for _, g in df.groupby(["__model", "__scenario", "strategy", "mode"], dropna=False):
        g = g.sort_values("round")
        lbl = series_label(g.iloc[0])
        color = pick_color(g.iloc[0]["mode"])
        marker = pick_marker(g.iloc[0]["strategy"])

        # Plot with markers
        ax.plot(
            g["round"].values,
            g[metric].values,
            label=lbl,
            marker=marker,
            linestyle='-',
            linewidth=1.8,
            markersize=6,
            color=color,
        )

    # Axis, title, legend
    ax.set_xlabel("Round")  # English: X-axis shows active learning rounds.
    ax.set_ylabel(metric.replace("_", " ").title())  # English: Y-axis shows the metric.
    ax.set_title(title)
    ax.grid(True, linewidth=0.5, alpha=0.4)

    # Place legend outside on the right for clarity
    ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=8)
    fig.tight_layout()

    # Save
    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    fig.savefig(outfile, dpi=200)
    plt.close(fig)


def pretty_scenario_name(s: str) -> str:
    """Turn an internal scenario token into a nice title snippet."""
    if s.upper() == "REAL":
        return "RATIO-REAL"
    # For tokens like '1-1-40', present as 'RATIO-1:1:40' in title
    return "RATIO-" + s.replace("-", ":")


def safe_filename(s: str) -> str:
    """Make a safe filename suffix from scenario (no colons)."""
    return s.replace(":", "-").replace("/", "_").replace("\\", "_")


# ---------------- Main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default=".", help="Root directory to scan")
    ap.add_argument("--pattern", type=str, default="round_log.csv", help="CSV filename pattern in subfolders")
    ap.add_argument("--out", type=str, default="./figs", help="Output directory for figures")
    ap.add_argument("--title-prefix", type=str, default="", help="Optional title prefix for the figures")
    ap.add_argument(
        "--include-substr",
        action="append",
        default=None,
        help="If set, only include files whose path contains ANY of these substrings (use multiple times).",
    )
    args = ap.parse_args()

    df = load_all_csvs(args.root, args.pattern)
    if df.empty:
        print("[INFO] No CSVs found. Nothing to plot.")
        return

    # Optional filtering by substrings
    if args.include_substr:
        mask = False
        for s in args.include_substr:
            mask = mask | df["__src"].str.contains(re.escape(s))
        df = df[mask]
        if df.empty:
            print("[INFO] After filtering, no CSVs remain. Nothing to plot.")
            return

    df = normalize_columns(df)

    # Determine scenarios present, keep a stable preferred order
    present = sorted(df["__scenario"].dropna().unique().tolist())
    # Prefer order: REAL first, then 1-1-40 (if present)
    order_pref = ["REAL", "1-1-40"]
    ordered_scenarios: List[str] = [s for s in order_pref if s in present] + [s for s in present if s not in order_pref]

    if not ordered_scenarios:
        print("[INFO] No scenarios detected.")
        return

    # For each scenario, make two figures
    for scen in ordered_scenarios:
        sub = df[df["__scenario"] == scen]
        if sub.empty:
            continue

        scen_title = pretty_scenario_name(scen)  # e.g., RATIO-REAL or RATIO-1:1:40
        scen_suffix = safe_filename(scen_title)  # safe for filenames

        t1 = ("{} - {} - Minority F1".format(args.title_prefix, scen_title)).strip(" -")
        t2 = ("{} - {} - Macro F1".format(args.title_prefix, scen_title)).strip(" -")

        out1 = os.path.join(args.out, f"minority_f1_{scen_suffix}.png")
        out2 = os.path.join(args.out, f"macro_f1_{scen_suffix}.png")

        plot_metric(sub, "minority_f1", t1, out1)
        plot_metric(sub, "macro_f1", t2, out2)

        print(f"[OK] Saved: {out1}")
        print(f"[OK] Saved: {out2}")

    print("[DONE] Per-scenario figures generated.")


if __name__ == "__main__":
    main()
