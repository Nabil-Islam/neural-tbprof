import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
from pathlib import Path
import pandas as pd

WORK = Path.cwd()
RESULTS = WORK / 'results'
METRICS = RESULTS / 'metrics'

who = pd.read_csv(METRICS / "who_by_drug_metrics.csv")
er = pd.read_csv(METRICS / "expert_rules_by_drug_metrics.csv")
filt = pd.read_csv(METRICS / "filtered_expert_rules_by_drug_metrics.csv")
tb = pd.read_csv(METRICS / "naiive_perf_metrics.csv",
                 # fix: skip_rows → skiprows
                 skiprows=1).rename(columns=str.lower)

drugs = who['drug'].tolist()
sen, sp = 'sensitivity', 'specificity'
f1 = 'f1'

wse, wsp = who[sen].tolist(),  who[sp].tolist()
ese, esp = er[sen].tolist(),   er[sp].tolist()
fse, fsp = filt[sen].tolist(), filt[sp].tolist()
wf1 = who[f1].tolist()
ef1 = er[f1].tolist()
ff1 = filt[f1].tolist()
# --- Parse naive CSV (assumed format per cell: "(val/bl_who,bl_er,bl_filt)") ---
raw_sen = tb[sen].tolist()
raw_sp = tb[sp].tolist()
raw_f1 = tb[f1].tolist()


def parse_val(s):
    return float(str(s).split('/')[0].strip('( '))


def parse_bl(s, idx):
    # fix: strip[')'] → strip(')')
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

# ── Style ────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size":         9,
    "axes.linewidth":    0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.major.size":  3,
    "ytick.major.size":  3,
    "xtick.direction":   "out",
    "ytick.direction":   "out",
    "pdf.fonttype":      42,
    "svg.fonttype":      "none",
})

# ── Plot config ───────────────────────────────────────────────────────────────
MODEL_NAMES = ["TB-Profiler", "WHO",      "Expert Rules", "Filtered"]
MARKERS = ["o",           "s",        "^",            "D"]
COLORS = ["#4878CF",     "#D65F5F",  "#6ACC65",      "#B47CC7"]
DOT_SIZE = 18
OFFSET_STEP = 0.14                          # horizontal jitter between models
offsets = np.array([-1.5, -0.5, 0.5, 1.5]) * OFFSET_STEP
DASH_HALF = 0.38                          # half-width of per-drug baseline tick

x = np.arange(len(drugs))

# ── Helper ────────────────────────────────────────────────────────────────────


def draw_dotplot(ax, series_4, baselines_3, ylabel, ylim_bottom=0.0):
    """
    series_4   : [tb, who, er, filt] — trained dots
    baselines_3: [who_bl, er_bl, filt_bl] — untrained lines, aligned to models 1-3
    """
    TICK_HALF = 0.07   # half-width of baseline tick, scoped to its sub-column

    # Dots (trained)
    for j, (vals, color, marker, name) in enumerate(
            zip(series_4, COLORS, MARKERS, MODEL_NAMES)):
        ax.scatter(x + offsets[j], vals,
                   color=color, marker=marker, s=DOT_SIZE,
                   zorder=4, label=name,
                   linewidths=0.5, edgecolors="white")

    # Untrained baseline — each aligned to its own model column (j=1,2,3)
    for j, (bl_vals, color) in enumerate(zip(baselines_3, COLORS[1:]), start=1):
        for i, bl in enumerate(bl_vals):
            cx = x[i] + offsets[j]   # centre on that model's sub-column
            ax.hlines(bl,
                      xmin=cx - TICK_HALF, xmax=cx + TICK_HALF,
                      colors=color, linewidths=1.4,
                      linestyles="--", zorder=5, alpha=0.7)

    # Faint vertical separators between drugs
    for i in range(len(drugs) - 1):
        ax.axvline(x=i + 0.5, color="#CCCCCC", linewidth=0.6,
                   linestyle="--", zorder=1)

    ax.set_xticks(x)
    ax.set_xticklabels(drugs, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_xlim(-0.7, len(drugs) - 0.3)
    ax.set_ylim(ylim_bottom, 1.10)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.1f}"))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, color="#EBEBEB", linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)


# ── Build figure ──────────────────────────────────────────────────────────────
for metric_vals, baselines, ylabel, fname, ylim_bottom in [
    ([tbse, wse, ese, fse], [wse_bl, ese_bl, fse_bl],
     "Sensitivity", "sensitivity", 0.0),
    ([tbsp, wsp, esp, fsp], [wsp_bl, esp_bl, fsp_bl],
     "Specificity", "specificity", 0.0),
    ([tbf1, wf1, ef1, ff1], [wf1_bl, ef1_bl, ff1_bl],
     "F1 Score",    "f1",           0.0),
]:
    fig, ax = plt.subplots(figsize=(7.0, 3.2))
    draw_dotplot(ax, metric_vals, baselines, ylabel, ylim_bottom=ylim_bottom)

    dot_handles = [
        mlines.Line2D([], [], color=c, marker=m, linestyle="None",
                      markersize=5, label=n,
                      markeredgecolor="white", markeredgewidth=0.5)
        for c, m, n in zip(COLORS, MARKERS, MODEL_NAMES)
    ]
    # Single style cue: dashed = untrained (no extra 3 entries)
    untrained_handle = mlines.Line2D([], [], color="#888888", linestyle="--",
                                     linewidth=1.2, label="Feed Forward")
    fig.legend(
        handles=dot_handles + [untrained_handle],
        loc="lower center", ncol=len(MODEL_NAMES) + 1,
        fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.08),
    )

    plt.tight_layout(rect=[0, 0.08, 1, 1])
    (RESULTS / "figures").mkdir(parents=True, exist_ok=True)
    fig.savefig(RESULTS / f"figures/{fname}.pdf", bbox_inches="tight")
    fig.savefig(RESULTS / f"figures/{fname}.svg", bbox_inches="tight")
    fig.savefig(RESULTS / f"figures/{fname}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

print("Saved sensitivity.* and specificity.* and f1.*")
