"""
Compare tb-profiler against the neural-network models, per drug.

Writes two tables:

  metrics/naiive_perf_metrics.csv
      All samples. tb-profiler + the naive (untrained, rule-weight) models.
      train.py / train_cr.py read tb-profiler's sensitivity from this file as
      their threshold target, so run comp.py once BEFORE training.

  metrics/test_perf_metrics.csv
      Held-out test runs only. tb-profiler + naive models + every trained
      model whose *_trained_calls.csv exists. Trained models can only be
      scored fairly on samples they weren't trained on, so every tool is
      restricted to the same test runs here. Run comp.py again AFTER training.
"""

import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

OLD_TB = Path(
    "/data/scratch/projects/punim1703/false_negative/old_tbprof/resources/data")
NN_RESULTS = Path("/data/scratch/projects/punim1703/nn_tb/REVISION/results")
METRICS = NN_RESULTS / "metrics"

# Naive model outputs: R/S calls for ALL samples, in vcf_list.txt row order.
NAIVE = {
    "who_nn": (METRICS / "who/who_predictions.csv", None),
    "er_nn":  (METRICS / "er/expert_rules_predictions.csv", None),
    # CR model emits [non-CR drugs, CR drugs]; reorder to alphabetical.
    # amk, bdq, cap, cfz, dlm, emb, eto, inh, kan, lfx, lzd, mfx, pza, rif, stp
    "cr_nn":  (METRICS / "cr/cross_expert_rules_predictions.csv",
               [0, 11, 1, 12, 2, 3, 14, 13, 4, 9, 5, 10, 6, 7, 8]),
}

# Trained model outputs: R/S calls for the TEST split only, with a 'run'
# column. Missing files are skipped, so this works before every model is trained.
TRAINED = {
    "who_trained":    METRICS / "who/who_trained_calls.csv",
    "er_trained":     METRICS / "er/expert_rules_trained_calls.csv",
    "filt_trained":   METRICS / "filt/filtered_expert_rules_trained_calls.csv",
    "who_cr_trained": METRICS / "cr/who_CR_trained_calls.csv",
    "er_cr_trained":  METRICS / "cr/expert_rules_CR_trained_calls.csv",
    "filt_cr_trained": METRICS / "cr/filtered_expert_rules_CR_trained_calls.csv",
}

# ---------------------------------------------------------------------------
truth = pd.read_csv(OLD_TB / "samples.csv", low_memory=False)
tb_prof = pd.read_csv(OLD_TB / "NEW_classifications.csv")
if "run" not in tb_prof.columns:
    raise ValueError("NEW_classifications.csv has no 'run' column; "
                     "can't restrict tb-profiler to the test runs")

with open(NN_RESULTS / "essentials/FINAL_SNP.json") as f:
    er_weights = json.load(f)
with open(NN_RESULTS / "essentials/who_snps.json") as f:
    who_weights = json.load(f)
if er_weights.keys() != who_weights.keys():
    raise ValueError("Weights keys don't match")

# e.g. 'Amikacin' -- prediction columns
drugs = list(er_weights.keys())
lower_drugs = [d.lower() for d in drugs]   # e.g. 'amikacin' -- truth columns
print(drugs)

runs = pd.read_csv(NN_RESULTS / "essentials/vcf_list.txt",
                   sep=" ", header=None)
runs.columns = ["run"]
runs["run"] = runs["run"].apply(lambda x: x.split("/")[-1].split(".")[0])


def load_naive(path, reorder=None):
    """Naive predictions have no run ids; rows follow vcf_list.txt order."""
    df = pd.read_csv(path)
    if reorder is not None:
        df = df.iloc[:, reorder]
    if df.shape[1] != len(drugs):
        raise ValueError(
            f"{path}: {df.shape[1]} columns, expected {len(drugs)}")
    if len(df) != len(runs):
        # The old index merge silently truncated here. A length mismatch means
        # the row -> run mapping can't be trusted, so stop instead.
        raise ValueError(
            f"{path}: {len(df)} rows but vcf_list.txt has {len(runs)}")
    df.columns = drugs
    return pd.concat([runs, df.reset_index(drop=True)], axis=1)


def load_trained(path):
    """Trained calls carry a 'run' column and lowercase drug columns."""
    df = pd.read_csv(path)
    if "run" not in df.columns:
        raise ValueError(f"{path} has no 'run' column; regenerate it with the "
                         "updated train script")
    rename = {d.lower(): d for d in drugs}
    missing = [c for c in rename if c not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing drug columns: {missing}")
    return df[["run", *rename]].rename(columns=rename)


def classify(pred_df):
    """Per (run, drug) TP/TN/FP/FN against samples.csv."""
    merged = pd.merge(pred_df, truth, on="run", how="inner")
    out = []
    for d, low in zip(drugs, lower_drugs):
        sub = merged[["run", d, low]].dropna(subset=[low])
        p, t = sub[d].to_numpy(), sub[low].to_numpy()
        cl = np.where(p == t,
                      np.where((p == "S") | (p == ""), "TN", "TP"),
                      np.where(p == "R", "FP", "FN"))
        out.append(pd.DataFrame({"run": sub["run"].to_numpy(), "drug": low,
                                 "classification": cl, "tool": "Illumina"}))
    return pd.concat(out, ignore_index=True)


def safe_div(n, d):
    return n / d if d != 0 else float("nan")


def fmt(x):
    return f"{x:.3f}" if not math.isnan(x) else "NA"


def summarise(classified, run_subset=None):
    """
    classified: {tool: classification table}, tb-profiler under key 'tb'.
    run_subset: restrict every tool to these runs (None = all samples).
    Returns (results_df, nn_tool_names).
    """
    if run_subset is not None:
        classified = {k: v[v["run"].isin(run_subset)]
                      for k, v in classified.items()}
    nn_tools = [k for k in classified if k != "tb"]

    results = []
    for drug in lower_drugs:
        mets = defaultdict(dict)
        for tool, table in classified.items():
            sub = table[table["drug"] == drug]["classification"]
            TP, TN = int((sub == "TP").sum()), int((sub == "TN").sum())
            FP, FN = int((sub == "FP").sum()), int((sub == "FN").sum())
            mets[tool]["sensitivity"] = safe_div(TP, TP + FN)
            mets[tool]["specificity"] = safe_div(TN, TN + FP)
            mets[tool]["precision"] = safe_div(TP, TP + FP)
            mets[tool]["f1"] = safe_div(2 * TP, 2 * TP + FP + FN)
            mets[tool]["n"] = TP + TN + FP + FN

        row = {"Drug": drug}
        for m in ["sensitivity", "specificity", "precision", "f1"]:
            inner = ", ".join(fmt(mets[t][m]) for t in nn_tools)
            row[m.capitalize()] = f"{fmt(mets['tb'][m])} / ({inner})"
        row["N"] = f"{mets['tb']['n']
                      } / ({', '.join(str(mets[t]['n']) for t in nn_tools)})"
        results.append(row)
    return pd.DataFrame(results), nn_tools


def write_table(path, df, nn_tools, note):
    title = f"#Metrics in form tb-profiler / ({', '.join(nn_tools)}){note}\n"
    with open(path, "w") as f:
        f.write(title)
        df.to_csv(f, index=False)
    print(f"Saved: {path}")
    print(df.to_string(index=False))


# --- classify everything ----------------------------------------------------
classified = {"tb": tb_prof}
for tool, (path, reorder) in NAIVE.items():
    classified[tool] = classify(load_naive(path, reorder))

# --- 1. all samples: tb-profiler + naive models -----------------------------
full_df, full_tools = summarise(classified)
write_table(METRICS / "naiive_perf_metrics.csv", full_df, full_tools,
            " -- all samples")

# --- 2. test runs only: everything, including trained models ----------------
trained = {k: load_trained(p) for k, p in TRAINED.items() if p.exists()}
if not trained:
    print("\nNo *_trained_calls.csv found yet; skipping the test-set comparison.")
else:
    run_sets = {k: frozenset(df["run"]) for k, df in trained.items()}
    test_runs = next(iter(run_sets.values()))
    if any(s != test_runs for s in run_sets.values()):
        sizes = {k: len(s) for k, s in run_sets.items()}
        raise ValueError(
            f"trained models were evaluated on different test runs {sizes}; "
            "they must share the same split (seed 42 + same labelled_vcfs.csv)")

    for tool, df in trained.items():
        classified[tool] = classify(df)
    print(f"\nTrained models found: {list(trained)}; "
          f"scoring all tools on {len(test_runs)} shared test runs")
    test_df, test_tools = summarise(classified, run_subset=test_runs)
    write_table(METRICS / "test_perf_metrics.csv", test_df, test_tools,
                f" -- {len(test_runs)} held-out test runs only")
