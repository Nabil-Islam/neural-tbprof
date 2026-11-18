"""
Decision-threshold selection: pick a probability cutoff per label that
meets a target sensitivity/specificity floor (e.g. matching or beating
an existing reference tool), falling back to the Youden-optimal point
if the floor is unreachable.
"""

import numpy as np
from sklearn.metrics import roc_curve

from tbml.confusion import confusion_stats


def pick_threshold(y_true, y_pred, target_sens, target_spec=0.98,
                    default_thr=0.5):
    """
    Choose a decision threshold for one label.

    Strategy:
      1. If the default threshold already clears both the target
         sensitivity and specificity, use it (keeps thresholds simple
         and interpretable where possible).
      2. Otherwise, sweep the ROC curve for thresholds that clear both
         floors and pick the one maximizing sensitivity among those.
      3. If no threshold clears both floors, fall back to the
         Youden-optimal threshold (max of tpr - fpr) and flag it as
         "unmatched" via the returned `matched` flag.

    Parameters
    ----------
    y_true, y_pred : 1-D arrays, already NaN-filtered, for a single label.
    target_sens : float
        Minimum sensitivity to aim for (e.g. from a reference tool).
    target_spec : float
        Minimum specificity to aim for.
    default_thr : float
        Threshold to try first before sweeping.

    Returns
    -------
    threshold : float
    matched : bool
        True if a threshold was found that clears both floors
        (whether via the default or the sweep). False if we had to
        fall back to the Youden point.
    """
    stats = confusion_stats(y_true, y_pred, default_thr)
    if stats["specificity"] >= target_spec and stats["sensitivity"] >= target_sens:
        return default_thr, True

    fpr, tpr, thr = roc_curve(y_true, y_pred)
    spec = 1 - fpr

    valid = (tpr >= target_sens) & (spec >= target_spec)
    if valid.any():
        idx = np.where(valid)[0][np.argmax(tpr[valid])]
        return float(thr[idx]), True

    # Floor unreachable at any threshold -> Youden-optimal fallback
    j = tpr - fpr
    return float(thr[np.argmax(j)]), False


def threshold_at_spec(y_true, y_pred, target_spec):
    """
    Smallest threshold whose specificity is at least `target_spec`, with
    calls made as y_pred > threshold (as in confusion_stats).

    Unlike pick_threshold there is no fallback: every label lands at the
    requested specificity (or the nearest achievable point above it), so
    sensitivities can be compared like-for-like against a reference tool
    run at that specificity.
    """
    y_true = np.asarray(y_true)
    neg = np.sort(np.asarray(y_pred)[y_true == 0])
    if len(neg) == 0:
        raise ValueError("no negatives: specificity is undefined")
    # need #(neg <= thr) >= ceil(target * n); small epsilon guards against
    # float error such as 0.98 * 100 = 98.00000000000001
    k = int(np.ceil(target_spec * len(neg) - 1e-9))
    k = min(max(k, 1), len(neg))
    return float(neg[k - 1])
