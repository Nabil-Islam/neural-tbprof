"""
Per-drug sensitivity / specificity / F1: TB-Profiler vs the WHO and
WHO + expert-rule models, each shown untrained (catalogue weights, hollow
marker) and trained (filled marker), joined by a grey stem.

Everything comes from results/metrics/test_perf_metrics.csv (comp_new.py),
so every tool is scored on the same held-out test isolates, with the trained
models at TB-Profiler's specificity. Columns are found by name from that
file's header line, so adding or removing tools in comp_new.py doesn't
shift them.

    python scripts/plot.py          # from the REVISION root, after comp_new.py
"""

import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RESULTS = Path.cwd() / "results"
METRICS_CSV = RESULTS / "metrics/test_perf_metrics.csv"
FIGURES = RESULTS / "figures"

# (label, untrained column, trained column, colour); None = no untrained anchor
MODELS = [
    ("TB-Profiler",        None,     "tb",          "#2E5E9E"),
    ("WHO",                "who_nn", "who_trained", "#C05A2E"),
    ("WHO + Expert Rules", "er_nn",  "er_trained",  "#3A7D44"),
]
METRICS = [("Sensitivity", "sensitivity"),
           ("Specificity", "specificity"),
           ("F1",          "f1")]


def load(path):
    """Return {tool: DataFrame(drug x metric)} parsed from comp_new.py's table."""
    with open(path) as f:
        header = f.readline()
    m = re.search(r"tb-profiler / \(([^)]*)\)", header)
    if not m:
        raise ValueError(f"can't find the tool list in the first line of {path}:\n{header}")
    tools = ["tb"] + [t.strip() for t in m.group(1).split(",")]

    df = pd.read_csv(path, skiprows=1)
    out = {t: pd.DataFrame(index=df["Drug"]) for t in tools}
    for col in ["Sensitivity", "Specificity", "Precision", "F1"]:
        for drug, cell in zip(df["Drug"], df[col]):
            first, rest = str(cell).split("/", 1)
            vals = [first] + rest.strip().strip("()").split(",")
            for t, v in zip(tools, vals):
                v = v.strip()
                out[t].loc[drug, col.lower()] = np.nan if v in ("NA", "nan", "") else float(v)
    return out


def style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Nimbus Sans", "Liberation Sans", "DejaVu Sans"],
        "font.size": 8,
        "axes.linewidth": 0.75,
        "xtick.major.width": 0.75, "ytick.major.width": 0.75,
        "xtick.major.size": 3.5, "ytick.major.size": 3.5,
        "xtick.direction": "out", "ytick.direction": "out",
        "axes.spines.top": False, "axes.spines.right": False,
        "pdf.fonttype": 42, "svg.fonttype": "none",
    })


DOT_SIZE = 18
ANCHOR_SIZE = 9
STEM_COLOR = "#C8C8C8"
OFFSET_STEP = 0.22


def draw(ax, data, metric, drugs, ylabel):
    x = np.arange(len(drugs))
    offsets = (np.arange(len(MODELS)) - (len(MODELS) - 1) / 2) * OFFSET_STEP

    for (label, untrained, trained, color), off in zip(MODELS, offsets):
        y_tr = data[trained].loc[drugs, metric].to_numpy(float)
        if untrained is not None:
            y_un = data[untrained].loc[drugs, metric].to_numpy(float)
            ax.vlines(x + off, y_un, y_tr, colors=STEM_COLOR, linewidth=0.8, zorder=2)
            ax.scatter(x + off, y_un, s=ANCHOR_SIZE, facecolors="white",
                       edgecolors=color, linewidths=0.8, zorder=3)
        ax.scatter(x + off, y_tr, s=DOT_SIZE, color=color, edgecolors="white",
                   linewidths=0.4, zorder=4, label=label)

    ax.set_xticks(x)
    ax.set_xticklabels([d.capitalize() for d in drugs], rotation=30, ha="right", fontsize=9)
    ax.set_xlim(-0.6, len(drugs) - 0.4)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel(ylabel, fontsize=10, fontweight="bold", labelpad=6)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.1f}"))
    ax.yaxis.grid(True, color="#EBEBEB", linestyle="--", linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)


def legend(fig):
    handles = [mlines.Line2D([], [], color=c, marker="o", linestyle="None", markersize=5,
                             markeredgecolor="white", markeredgewidth=0.4, label=label)
               for label, _, _, c in MODELS]
    handles.append(mlines.Line2D([], [], color="white", marker="o", linestyle="None",
                                 markersize=4, markeredgecolor="#555555",
                                 markeredgewidth=0.8, label="Untrained (catalogue weights)"))
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=8.5,
               frameon=False, bbox_to_anchor=(0.5, -0.12))


def main():
    style()
    data = load(METRICS_CSV)
    needed = {c for _, u, t, _ in MODELS for c in (u, t) if c}
    missing = needed - set(data)
    if missing:
        raise SystemExit(f"{METRICS_CSV} has no columns for {sorted(missing)}; "
                         f"run train_new.py for those models, then comp_new.py")
    drugs = list(data["tb"].index)
    FIGURES.mkdir(parents=True, exist_ok=True)

    for ylabel, metric in METRICS:
        fig, ax = plt.subplots(figsize=(7.5, 3.5))
        draw(ax, data, metric, drugs, ylabel)
        ax.set_title(f"{ylabel} on held-out test isolates", fontsize=11,
                     fontweight="bold", pad=10)
        legend(fig)
        fig.tight_layout()
        out = FIGURES / f"{metric}.png"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {out}")


if __name__ == "__main__":
    main()
