"""
Turning model probability scores into S/R calls using per-drug thresholds.

No tensorflow dependency.
"""

import numpy as np
import pandas as pd


def scores_to_calls(y_pred, drugs, thresholds, default_thr=0.5,
                    pos='R', neg='S'):
    """
    Convert a (n_samples, n_drugs) score matrix to 'R'/'S' calls.

    A sample is called resistant when score > threshold for that drug
    (strict >, matching how sensitivity/specificity are computed in
    tbml.confusion.confusion_stats, so the calls agree with the metrics).

    Parameters
    ----------
    y_pred : array, shape (n_samples, n_drugs)
    drugs : list of str
        Column names, in the same order as y_pred's columns.
    thresholds : dict
        drug -> threshold. Drugs missing from it fall back to default_thr.
    """
    y_pred = np.asarray(y_pred)
    thr = np.array([thresholds.get(d, default_thr) for d in drugs])
    calls = np.where(y_pred > thr[None, :], pos, neg)
    return pd.DataFrame(calls, columns=drugs)


def save_prediction_tables(y_pred, drugs, thresholds, scores_path, calls_path,
                           run_ids=None, default_thr=0.5):
    """
    Write two CSVs with identical layout (one row per sample, one column
    per drug, optional leading 'run' column):

      scores_path : raw sigmoid probabilities
      calls_path  : 'R' / 'S' using the per-drug thresholds

    Returns (scores_df, calls_df).
    """
    scores_df = pd.DataFrame(np.asarray(y_pred), columns=drugs)
    calls_df = scores_to_calls(y_pred, drugs, thresholds, default_thr)

    if run_ids is not None:
        run_ids = np.asarray(run_ids)
        if len(run_ids) != len(scores_df):
            raise ValueError(
                f"{len(run_ids)} run ids for {len(scores_df)} prediction rows"
            )
        scores_df.insert(0, 'run', run_ids)
        calls_df.insert(0, 'run', run_ids)

    scores_df.to_csv(str(scores_path), index=False)
    calls_df.to_csv(str(calls_path), index=False)
    return scores_df, calls_df
