"""
Plotting helpers: shared matplotlib/seaborn style, learning-curve panels,
and confusion-matrix grids.
"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def set_plot_style():
    """Apply the shared matplotlib/seaborn style used across figures."""
    matplotlib.use('Agg')
    sns.set_palette("muted")
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


def plot_learning_curves(epochs, curves, save_path_base, title,
                          test_values=None, figsize=(16, 4)):
    """
    Plot a row of 3 panels (loss, AUC, F1) over epochs. Handles both the
    "global" case (train vs val vs test line) and the "per-drug" case
    (single val line, no train/test) via optional keys.

    Parameters
    ----------
    epochs : sequence of epoch numbers (x-axis), shared across panels.
    curves : dict
        Keys among {"loss", "auc", "f1"} -> dict of line-name -> values,
        e.g. {"loss": {"train": [...], "val": [...]}, "auc": {...}, "f1": {...}}
        Line names starting with "test" are drawn as a horizontal dashed
        reference line at test_values[metric] instead of a curve.
    save_path_base : str or Path
        Path *without* extension; both .png and .pdf are written.
    title : str
        Figure suptitle.
    test_values : dict, optional
        Metric name -> scalar test value, drawn as a red dotted hline.
    figsize : tuple
    """
    metrics = ["loss", "auc", "f1"]
    labels = {"loss": "Loss", "auc": "AUC", "f1": "F1"}
    colors = {"train": "black", "val": "blue"}

    fig, axes = plt.subplots(1, len(metrics), figsize=figsize)
    if len(metrics) == 1:
        axes = [axes]

    for ax, metric in zip(axes, metrics):
        lines = curves.get(metric, {})
        for line_name, values in lines.items():
            style = "--" if line_name == "val" else "-"
            color = colors.get(line_name, None)
            ax.plot(epochs, values, color=color, linestyle=style,
                    label=line_name.capitalize())

        if test_values and metric in test_values:
            tv = test_values[metric]
            ax.axhline(y=tv, color='r', linestyle=':', label=f'Test ({tv:.4f})')

        ax.set_xlabel('Epoch', fontweight='bold')
        ax.set_ylabel(labels[metric], fontweight='bold',
                      rotation=0, labelpad=15, va='center')
        ax.set_title(labels[metric], fontweight='bold')
        ax.legend(frameon=False)

    fig.suptitle(title, fontweight='bold')
    fig.tight_layout()
    fig.savefig(f'{save_path_base}.png', dpi=150)
    fig.savefig(f'{save_path_base}.pdf', format='pdf')
    plt.close(fig)


def plot_confusion_grid(labels, stats_by_label, save_path_base, suptitle,
                         ncols=5, figsize=(18, 10),
                         class_names=("S", "R")):
    """
    Plot a grid of 2x2 confusion matrices, one per label.

    Parameters
    ----------
    labels : list of str
        Label names, used for subplot titles and to look up stats_by_label.
    stats_by_label : dict
        label -> dict with keys tn, fp, fn, tp, sensitivity, specificity
        (as returned by tbml.confusion.confusion_stats), or None to skip
        that label (e.g. insufficient data).
    save_path_base : str or Path
        Path without extension; both .png and .pdf are written.
    suptitle : str
    class_names : tuple of 2 str
        Names for the negative/positive class, e.g. ("S", "R") for
        Susceptible/Resistant.
    """
    nrows = -(-len(labels) // ncols)  # ceil div
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = np.array(axes).flatten()

    neg, pos = class_names

    for ax, label in zip(axes, labels):
        stats = stats_by_label.get(label)
        if stats is None:
            ax.axis('off')
            continue

        cm = np.array([[stats["tn"], stats["fp"]],
                       [stats["fn"], stats["tp"]]])
        ax.imshow(cm, cmap='Blues', vmin=cm.min(), vmax=cm.max())
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels([f'Pred {neg}', f'Pred {pos}'], fontweight='bold')
        ax.set_yticklabels([f'True {neg}', f'True {pos}'], fontweight='bold')
        ax.set_title(label.capitalize(), fontsize=9, fontweight='bold')

        thr = cm.max() / 2.0
        for r in range(cm.shape[0]):
            for c in range(cm.shape[1]):
                color = 'white' if cm[r, c] > thr else 'black'
                if r == 0 and c == 0:
                    dtext = f"{cm[r, c]}\n(Spec.={round(stats['specificity'], 3)})"
                elif r == 1 and c == 1:
                    dtext = f"{cm[r, c]}\n(Sens.={round(stats['sensitivity'], 3)})"
                else:
                    dtext = f"{cm[r, c]}"
                ax.text(c, r, dtext, ha='center', va='center',
                        color=color, fontsize=10, fontweight='bold')

        for side in ['top', 'bottom', 'left', 'right']:
            ax.spines[side].set_visible(True)
            ax.spines[side].set_linewidth(1.2)
            ax.spines[side].set_color('black')

    # turn off any unused trailing axes
    for ax in axes[len(labels):]:
        ax.axis('off')

    fig.suptitle(suptitle, fontweight='bold', fontsize=12)
    fig.tight_layout()
    fig.savefig(f'{save_path_base}.png', dpi=150)
    fig.savefig(f'{save_path_base}.pdf', bbox_inches='tight')
    plt.close(fig)
