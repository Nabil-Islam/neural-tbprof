import pandas as pd
import math
import json
from pathlib import Path
from collections import defaultdict

OLD_TB = Path(
    "/data/scratch/projects/punim1703/false_negative/old_tbprof/resources/data")
NN_RESULTS = Path("/data/scratch/projects/punim1703/nn_tb/REVISION/results")
RERUN = Path("/data/scratch/projects/punim1703/false_negative/tbprof-rerun")

truth = pd.read_csv(OLD_TB / "samples.csv")
tb_prof = pd.read_csv(OLD_TB / "NEW_classifications.csv")
# classifications already done

with open(NN_RESULTS / "FINAL_SNP.json") as f:
    er_weights = json.load(f)
er_nn = pd.read_csv(NN_RESULTS / "er_predictions.csv")

with open(NN_RESULTS / "who_snps.json") as f:
    who_weights = json.load(f)
who_nn = pd.read_csv(NN_RESULTS / "who_predictions.csv")

if er_weights.keys() != who_weights.keys():
    raise ValueError("Weights keys don't match")
weights = er_weights

drugs = list(weights.keys())
lower_drugs = [d.lower() for d in drugs]

assert len(drugs) == who_nn.shape[1], (
    f"Drug count mismatch: {len(drugs)} drugs vs {who_nn.shape[1]} columns"
)
assert len(drugs) == er_nn.shape[1], (
    f"Drug count mismatch: {len(drugs)} drugs vs {er_nn.shape[1]} columns"
)
who_nn.columns = drugs
er_nn.columns = drugs


# load neural network runs
runs = pd.read_csv(NN_RESULTS / 'vcf_list.txt', sep=" ", header=None)
runs.columns = ['run']
runs['run'] = runs['run'].apply(lambda x: x.split("/")[-1].split(".")[0])
who_nn = pd.merge(runs, who_nn, left_index=True, right_index=True, how='inner')
er_nn = pd.merge(runs, er_nn, left_index=True, right_index=True, how='inner')

who_nn_class = pd.merge(who_nn, truth, on='run', how='inner')
er_nn_class = pd.merge(er_nn, truth, on='run', how='inner')

cl_dt = {}
for name, df in [("who_nn_class", who_nn_class), ("er_nn_class", er_nn_class)]:
    nn_rows = []
    for drug in lower_drugs:
        for _, row in df.iterrows():
            r = row['run']
            t = row[drug]
            if pd.isna(t):
                continue
            p = row[drug.capitalize()]
            if p == t:
                cl = 'TN' if p == 'S' or p == '' else 'TP'
            else:
                cl = 'FP' if p == 'R' else 'FN'
            nn_rows.append(
                {'run': r, 'drug': drug, 'classification': cl, 'tool': 'Illumina'})
    cl_dt[name] = pd.DataFrame(nn_rows)

who_nn_flt = cl_dt["who_nn_class"]
er_nn_flt = cl_dt["er_nn_class"]

results = []
for drug in lower_drugs:
    who_nn_subset = who_nn_flt[who_nn_flt['drug'] == drug]
    er_nn_subset = er_nn_flt[er_nn_flt['drug'] == drug]
    tb_prof_subset = tb_prof[tb_prof['drug'] == drug]
    subsets = {'who_nn': who_nn_subset,
               'er_nn': er_nn_subset, 'tb': tb_prof_subset}
    cnts = defaultdict(dict)
    for cl in ['TN', 'TP', 'FN', 'FP']:
        for tool in ['who_nn', 'er_nn', 'tb']:
            cnts[tool][cl] = int(subsets[tool]['classification'].eq(cl).sum())
    mets = defaultdict(dict)

    def safe_div(n, d):
        return n / d if d != 0 else float('nan')
    for tool in ['who_nn', 'er_nn', 'tb']:
        TP = cnts[tool]['TP']
        TN = cnts[tool]['TN']
        FP = cnts[tool]['FP']
        FN = cnts[tool]['FN']
        sensitivity = safe_div(TP, TP + FN)
        specificity = safe_div(TN, TN + FP)
        precision = safe_div(TP, TP + FP)
        accuracy = safe_div(TP + TN, TP + TN + FP + FN)
        mets[tool]['sensitivity'] = sensitivity
        mets[tool]['specificity'] = specificity
        mets[tool]['accuracy'] = accuracy
        mets[tool]['balanced_accuracy'] = (specificity + sensitivity) / 2
        mets[tool]['precision'] = precision
        mets[tool]['f1'] = safe_div(2 * TP, (2 * TP + FP + FN))

    def na_format(tb, who, er):
        def fmt(x):
            return f"{x:.3f}" if not math.isnan(x) else "NA"
        return f"{fmt(tb)} / ({fmt(who)}, {fmt(er)})"
    tb = mets['tb']
    who = mets['who_nn']
    er = mets['er_nn']
    results.append({
        "Drug": drug,
        "Sensitivity": na_format(tb['sensitivity'], who['sensitivity'], er['sensitivity']),
        "Specificity": na_format(tb['specificity'], who['specificity'], er['specificity']),
        "Accuracy": na_format(tb['accuracy'], who['accuracy'], er['accuracy']),
        "Balanced_Accuracy": na_format(tb['balanced_accuracy'], who['balanced_accuracy'], er['balanced_accuracy']),
        "Precision": na_format(tb['precision'], who['precision'], er['precision']),
        "F1": na_format(tb['f1'], who['f1'], er['f1'])
    })
results_df = pd.DataFrame(results)
title = f'#Metrics in form tb-profiler / (WHO snps NN, expert rules NN)\n'
with open(NN_RESULTS / "perf_metrics.csv", "w") as f:
    f.write(title)
    results_df.to_csv(f, index=False)
