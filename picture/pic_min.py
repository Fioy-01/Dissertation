# -*- coding: utf-8 -*-
"""
Active Learning Learning-Curves Plotter (2 models in one figure)
- Scenarios: 1) RATIO-1-1-40, 2) RATIO-REAL
- Lines: {BERT, DEBERTA} × {Random, Entropy, ALVIN, Short-Simple, Long-Simple}
- Metric: Minority-F1 only
- Colors by MODEL; Markers by STRATEGY
- Legend placed OUTSIDE on the right
- X-axis cropped from 2200 with extra left padding; Y-axis has dynamic padding

All plotting comments are in English as requested.
"""
import os, re, glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ---------------- Config ----------------
ROOT = "."                 # root folder that contains subfolders
OUTPUT_DIR = "./figs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Only BERT and DEBERTA
MODELS_TO_PLOT = ["BERT", "DEBERTA"]
STRATEGIES_TO_PLOT = ["random", "entropy", "alvin", "short_simple", "long_simple"]

# Axis settings
MIN_X = 2200
X_PAD = 20
Y_PAD_FRAC = 0.05
XTICKS = [2200, 2400, 2600, 2800]

# Colors by MODEL
MODEL_COLOR = {
    "BERT":    "#d62728",  # red
    "DEBERTA": "#1f77b4",  # blue
}

# Markers by STRATEGY
STRAT_MARKER = {
    "random": "^",
    "entropy": "s",
    "alvin":   "o",
    "short_simple": "D",
    "long_simple":  "P",
}

# Legend labels
STRAT_LABEL = {
    "random": "Random",
    "entropy": "Uncertainty",
    "alvin": "ALVIN",
    "short_simple": "Short-Simple (Gen.)",
    "long_simple":  "Long-Simple (Gen.)"
}

# ---------------- Helpers ----------------
def find_csv_files(root):
    return glob.glob(os.path.join(root, "**", "*.csv"), recursive=True)

def parse_meta_from_path(path):
    folder = os.path.basename(os.path.dirname(path))
    pat = (
        r"(?P<model>BERT|DEBERTA|ROBERTA|DISTILBERT)"
        r"_RATIO-(?P<scenario>REAL|1-1-40)"
        r"_(?P<strategy>random|entropy|alvin|short_simple|long_simple)_uncontrolled"
    )
    m = re.match(pat, folder, flags=re.IGNORECASE)
    if not m:
        fname = os.path.basename(path)
        m = re.search(pat, fname, flags=re.IGNORECASE)
    if not m:
        return None
    g = m.groupdict()
    return g["model"].upper(), g["scenario"].upper(), g["strategy"].lower()

def load_one_csv(csv_path):
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        df = pd.read_csv(csv_path, sep=";")
    lower_map = {c.lower(): c for c in df.columns}
    ren = {}
    want = {
        "strategy": ["strategy"],
        "mode": ["mode"],
        "round": ["round","iter","iteration"],
        "labeled_total": ["labeled_total","labeled","labeledsize","labeled_count"],
        "minority_f1": ["minority_f1","minority-f1","f1_minority","minorityf1"],
    }
    for std, variants in want.items():
        for v in variants:
            if v in lower_map:
                ren[lower_map[v]] = std
                break
    df = df.rename(columns=ren)
    for col in ["labeled_total","round","minority_f1"]:
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "minority_f1" in df and df["minority_f1"].dropna().gt(1.5).mean() > 0.5:
        df["minority_f1"] = df["minority_f1"] / 100.0
    return df

def sem(x):
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) == 0: return np.nan
    return x.std(ddof=1) / np.sqrt(len(x))

def aggregate_runs(df, metric="minority_f1"):
    grp = df.groupby(["scenario","model","strategy","labeled_total"], as_index=False).agg(
        mean_metric=(metric, "mean"),
        n=("run_id", "nunique")
    )
    se = df.groupby(["scenario","model","strategy","labeled_total"])[metric].apply(sem).reset_index(name="sem")
    grp = grp.merge(se, on=["scenario","model","strategy","labeled_total"], how="left")
    grp["ci95"] = 1.96 * grp["sem"]
    return grp.sort_values(["labeled_total"])

def plot_scenario(df_agg, scenario, metric="minority_f1"):
    fig, ax = plt.subplots(figsize=(8.5, 5.0), dpi=150)
    ylabel = "Minority-F1"

    plotted_parts = []
    for model in MODELS_TO_PLOT:
        for strat in STRATEGIES_TO_PLOT:
            sub = df_agg[
                (df_agg["scenario"] == scenario) &
                (df_agg["model"] == model) &
                (df_agg["strategy"] == strat) &
                ((df_agg["labeled_total"] >= MIN_X) if MIN_X is not None else True)
            ].sort_values("labeled_total")
            if sub.empty:
                continue
            plotted_parts.append(sub[["labeled_total","mean_metric","ci95"]])
            ax.plot(sub["labeled_total"], sub["mean_metric"],
                    marker=STRAT_MARKER[strat], linestyle="-",
                    linewidth=1.8, markersize=6.5,
                    label=f"{model}–{STRAT_LABEL[strat]}",
                    color=MODEL_COLOR[model])
            if sub["ci95"].notna().any():
                ax.fill_between(sub["labeled_total"],
                                (sub["mean_metric"] - sub["ci95"]).clip(lower=0),
                                (sub["mean_metric"] + sub["ci95"]).clip(upper=1),
                                alpha=0.10, linewidth=0,
                                color=MODEL_COLOR[model])

    ax.set_xlabel("Labeled Total")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{scenario.replace('-', ':')} — {ylabel}")
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.5)
    ax.set_axisbelow(True)

    if plotted_parts:
        allp = pd.concat(plotted_parts, ignore_index=True)
        if MIN_X is not None:
            x_left = max(0, MIN_X - X_PAD)
        else:
            x_left = max(0, allp["labeled_total"].min() - X_PAD)
        x_right = allp["labeled_total"].max() + 0.01
        ax.set_xlim(x_left, x_right)
        ax.set_xticks([t for t in XTICKS if x_left <= t <= x_right])

        y_min = float(allp["mean_metric"].min())
        y_max = float(allp["mean_metric"].max())
        y_span = max(1e-6, y_max - y_min)
        pad = max(0.02, Y_PAD_FRAC * y_span)
        ax.set_ylim(max(0.0, y_min - pad), min(1.0, y_max + pad))
    else:
        ax.set_ylim(0.0, 1.0)

    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5),
              fontsize=9, frameon=False, handlelength=2.0, labelspacing=0.6)

    fig.tight_layout(rect=[0, 0, 0.80, 1])

    model_tag = "_".join(MODELS_TO_PLOT)
    out = os.path.join(OUTPUT_DIR, f"fig_learning_curve_{model_tag}_{scenario}_minorityF1.png")
    fig.savefig(out)
    print(f"[Saved] {out}")

# ---------------- Load & Combine ----------------
rows = []
for csv in find_csv_files(ROOT):
    meta = parse_meta_from_path(csv)
    if not meta:
        continue
    model, scenario, strategy = meta
    if model not in tuple(MODELS_TO_PLOT):
        continue
    if strategy not in set(STRATEGIES_TO_PLOT):
        continue
    df = load_one_csv(csv)
    if "labeled_total" not in df or "minority_f1" not in df:
        continue
    df = df.assign(model=model, scenario=scenario, strategy=strategy, run_id=os.path.basename(csv))
    keep_cols = ["scenario","model","strategy","labeled_total","minority_f1","run_id"]
    df = df[[c for c in keep_cols if c in df.columns]]
    rows.append(df)

if not rows:
    raise SystemExit("No valid CSV files found. Check ROOT path and folder naming pattern.")

data = pd.concat(rows, ignore_index=True)

# ---------------- Aggregate & Plot ----------------
metric = "minority_f1"
agg = aggregate_runs(data.dropna(subset=[metric]), metric=metric)
for scen in ["1-1-40", "REAL"]:
    if (agg["scenario"] == scen).any():
        plot_scenario(agg, scen, metric=metric)

# ---------------- Export aggregated table ----------------
out_csv = os.path.join(OUTPUT_DIR, f"learning_curves_agg_minorityF1_{'_'.join(MODELS_TO_PLOT)}.csv")
aggregate_runs(data.dropna(subset=["minority_f1"]), "minority_f1").to_csv(out_csv, index=False)
print(f"[Saved] {out_csv}")
