import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
from pathlib import Path
import pandas as pd

WORK = Path.cwd()
RESULTS = WORK / 'results'
METRICS = RESULTS / 'metrics'

who = pd.read_csv(METRICS / "who/who_by_drug_metrics.csv")
er = pd.read_csv(METRICS / "er/expert_rules_by_drug_metrics.csv")
filt = pd.read_csv(
    METRICS / "filt/filtered_expert_rules_by_drug_metrics.csv")
tb = pd.read_csv(METRICS / "naiive_perf_metrics.csv",
                 skiprows=1).rename(columns=str.lower)

drugs_list = who['drug'].tolist()
sen, sp = 'sensitivity', 'specificity'
f1 = 'f1'

wse, wsp = who[sen].tolist(),  who[sp].tolist()
ese, esp = er[sen].tolist(),   er[sp].tolist()
fse, fsp = filt[sen].tolist(), filt[sp].tolist()
wf1 = who[f1].tolist()
ef1 = er[f1].tolist()
ff1 = filt[f1].tolist()

raw_sen = tb[sen].tolist()
raw_sp = tb[sp].tolist()
raw_f1 = tb[f1].tolist()


def parse_val(s):
    return float(str(s).split('/')[0].strip('( '))


def parse_bl(s, idx):
    return float(str(s).split('/')[1].strip('() ').split(',')[idx].strip())


tbse = [parse_val(s) for s in raw_sen]
tbsp = [parse_val(s) for s in raw_sp]
tbf1 = [parse_val(s) for s in raw_f1]

wse_bl = [parse_bl(s, 0) for s in raw_sen]
ese_bl = [parse_bl(s, 1) for s in raw_sen]
fse_bl = [parse_bl(s, 2) for s in raw_sen]

wsp_bl = [parse_bl(s, 0) for s in raw_sp]
esp_bl = [parse_bl(s, 1) for s in raw_sp]
fsp_bl = [parse_bl(s, 2) for s in raw_sp]

wf1_bl = [parse_bl(s, 0) for s in raw_f1]
ef1_bl = [parse_bl(s, 1) for s in raw_f1]
ff1_bl = [parse_bl(s, 2) for s in raw_f1]

plt.rcParams.update({
    "font.family":       "sans-serif",
    "font.sans-serif":   ["Nimbus Sans", "Liberation Sans", "DejaVu Sans"],
    "font.size":         8,
    "axes.linewidth":    0.75,
    "xtick.major.width": 0.75,
    "ytick.major.width": 0.75,
    "xtick.major.size":  3.5,
    "ytick.major.size":  3.5,
    "xtick.direction":   "out",
    "ytick.direction":   "out",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "pdf.fonttype":      42,
    "svg.fonttype":      "none",
})

MODEL_NAMES = ["TB-Profiler", "WHO",
               "WHO+Expert Rules", "WHO+Expert Rules+Filtered"]
MARKERS = ["o", "o", "o", "o"]
COLORS = ["#2E5E9E", "#C05A2E", "#3A7D44", "#8B3A62"]

DOT_SIZE = 16          # Scaled down to prevent bleeding
STEM_COLOR = "#DCDCDC"  # Muted, light gray to prevent error bar confusion
OFFSET_STEP = 0.10     # Reduced from 0.16 to pull columns closer together
offsets = np.array([-1.5, -0.5, 0.5, 1.5]) * OFFSET_STEP

x = np.arange(len(drugs_list))


def draw_dotplot(ax, series_4, baselines_3, ylabel, target_drugs, ylim_bottom=0.0):
    """
    Renders an academic grouped lollipop scatter plot with distinct baseline anchors.
    """
    for j, bl_vals in enumerate(baselines_3, start=1):
        trained_vals = series_4[j]
        for i in range(len(target_drugs)):
            cx = x[i] + offsets[j]
            # Draw crisp thin vertical tracker between baseline configuration and final output
            ax.vlines(cx, ymin=bl_vals[i], ymax=trained_vals[i],
                      colors=STEM_COLOR, linewidth=0.7, zorder=2)

    for j, (vals, color, marker, name) in enumerate(zip(series_4, COLORS, MARKERS, MODEL_NAMES)):
        ax.scatter(x + offsets[j], vals,
                   color=color, marker=marker, s=DOT_SIZE,
                   zorder=4, label=name,
                   linewidths=0.4, edgecolors="white")

    for j, (bl_vals, color) in enumerate(zip(baselines_3, COLORS[1:]), start=1):
        ax.scatter(x + offsets[j], bl_vals,
                   color='white', marker='o', s=5,  # Reduced from 8 to match dot size
                   edgecolors=color, linewidths=0.7, zorder=3)

    display_labels = [str(d).capitalize() for d in target_drugs]
    ax.set_xticks(x)
    ax.set_xticklabels(display_labels, rotation=30, ha="right", fontsize=9.5)
    ax.set_ylabel(ylabel, fontsize=10, fontweight='bold', labelpad=6)
    ax.set_xlim(-0.6, len(target_drugs) - 0.4)
    ax.set_ylim(ylim_bottom, 1.05)

    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.1f}"))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Horizontal grid matrix markers (No messy vertical crosshatches)
    ax.yaxis.grid(True, color="#EBEBEB", linestyle="--",
                  linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)


for metric_vals, baselines, ylabel, fname, ylim_bottom in [
    ([tbse, wse, ese, fse], [wse_bl, ese_bl, fse_bl],
     "Sensitivity", "sensitivity", 0.0),
    ([tbsp, wsp, esp, fsp], [wsp_bl, esp_bl, fsp_bl],
     "Specificity", "specificity", 0.0),
    ([tbf1, wf1, ef1, ff1], [wf1_bl, ef1_bl, ff1_bl],
     "F1 Score",    "f1",           0.0),
]:
    fig, ax = plt.subplots(figsize=(7.5, 3.5))
    draw_dotplot(ax, metric_vals, baselines, ylabel,
                 drugs_list, ylim_bottom=ylim_bottom)

    dot_handles = [
        mlines.Line2D([], [], color=c, marker=m, linestyle="None",
                      markersize=4.5, label=n,  # Slightly scaled down for legend consistency
                      markeredgecolor="white", markeredgewidth=0.4)
        for c, m, n in zip(COLORS, MARKERS, MODEL_NAMES)
    ]

    untrained_handle = mlines.Line2D([], [], color="white", marker='o', linestyle="None",
                                     markersize=3.5, markeredgecolor="#555555", markeredgewidth=0.7,
                                     label="Untrained Baseline Anchor")

    fig.legend(
        handles=dot_handles + [untrained_handle],
        loc="lower center", ncol=3,
        fontsize=8.5, frameon=True, edgecolor='none', facecolor='white',
        bbox_to_anchor=(0.5, -0.15),
    )

    ax.set_title(f"Model Performance Evaluation: {
                 ylabel}", fontsize=11, fontweight='bold', pad=12)

    plt.tight_layout()

    (RESULTS / "figures").mkdir(parents=True, exist_ok=True)
    fig.savefig(RESULTS / f"figures/{fname}.png",
                dpi=300, bbox_inches="tight")
    plt.close(fig)

print("Process Complete. High-resolution compact layouts compiled successfully.")
