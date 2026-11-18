"""
TB-Profiler reference specificities: the per-drug operating point the
neural models are held to, so their sensitivity can be compared with
TB-Profiler's like-for-like.

Two sources:

  from_runs : computed from TB-Profiler's per-isolate TN/FP lists
              (NEW_updated_runs.json), restricted to the isolates you pass,
              e.g. the non-test isolates. Preferred: the target then never
              uses the test set.
  from_csv  : read from metrics/naiive_perf_metrics.csv (all samples,
              test included). Fallback only.

No tensorflow dependency.
"""

import json

import numpy as np
import pandas as pd


def from_runs(runs_json, keep_runs, drugs):
    """
    Specificity of TB-Profiler per drug = TN / (TN + FP), counting only
    isolates in `keep_runs`.
    """
    with open(runs_json) as f:
        rd = json.load(f)
    keep = set(map(str, keep_runs))

    def lower_keys(d):
        return {k.lower(): v for k, v in d.items()}
    tn_all, fp_all = lower_keys(rd.get('TN', {})), lower_keys(rd.get('FP', {}))

    out = {}
    for d in drugs:
        tn = sum(r in keep for r in tn_all.get(d.lower(), []))
        fp = sum(r in keep for r in fp_all.get(d.lower(), []))
        if tn + fp == 0:
            raise ValueError(f"no TB-Profiler TN/FP isolates for {d} among the kept runs")
        out[d] = tn / (tn + fp)
    return out


def from_csv(path, drugs, column='specificity'):
    """Parse the 'tb-profiler / (...)' cells of naiive_perf_metrics.csv."""
    tb = pd.read_csv(path, skiprows=1).rename(columns=str.lower)
    vals = {str(d).lower(): float(str(s).split('/')[0].strip('( '))
            for d, s in zip(tb['drug'], tb[column])}
    missing = [d for d in drugs if d.lower() not in vals]
    if missing:
        raise ValueError(f"no TB-Profiler {column} for {missing} in {path}")
    return {d: vals[d.lower()] for d in drugs}


def reference_specificity(drugs, keep_runs, runs_json=None, csv_path=None):
    """Use from_runs if runs_json is given, else from_csv. Returns (dict, source)."""
    if runs_json is not None:
        return from_runs(runs_json, keep_runs, drugs), f"{runs_json} (non-test isolates)"
    if csv_path is not None:
        print("WARNING: TB-Profiler specificity taken from all samples (test included); "
              "pass the TN/FP runs JSON to use non-test isolates only")
        return from_csv(csv_path, drugs), f"{csv_path} (all samples)"
    raise ValueError("need runs_json or csv_path")
