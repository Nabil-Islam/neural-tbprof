"""
Plain-numpy confusion-matrix statistics. Deliberately has no
tensorflow/keras dependency, so it (and anything that only needs it,
like tbml.thresholds) can be imported in lightweight environments.
"""

import numpy as np


def confusion_stats(y_true, y_pred, threshold, eps=1e-8):
    """
    Compute tp/fp/tn/fn and derived sensitivity/specificity/precision/f1
    at a given threshold. y_true and y_pred should already be NaN-filtered
    (1-D arrays for a single label).

    Returns a dict with keys: tp, fp, tn, fn, sensitivity, specificity,
    precision, f1.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    yb = (y_pred > threshold).astype(float)
    tp = ((yb == 1) & (y_true == 1)).sum()
    fn = ((yb == 0) & (y_true == 1)).sum()
    fp = ((yb == 1) & (y_true == 0)).sum()
    tn = ((yb == 0) & (y_true == 0)).sum()

    sens = tp / (tp + fn + eps)
    spec = tn / (tn + fp + eps)
    prec = tp / (tp + fp + eps)
    f1 = 2 * (prec * sens) / (prec + sens + eps)

    return {
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "sensitivity": sens, "specificity": spec,
        "precision": prec, "f1": f1,
    }
