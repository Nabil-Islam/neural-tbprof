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

with open(NN_RESULTS / "essentials/FINAL_SNP.json") as f:
    er_weights = json.load(f)
er_nn = pd.read_csv(NN_RESULTS / "metrics/er/expert_rules_predictions.csv")

with open(NN_RESULTS / "essentials/who_snps.json") as f:
    who_weights = json.load(f)
who_nn = pd.read_csv(NN_RESULTS / "metrics/who/who_predictions.csv")

filter_nn = pd.read_csv(
    NN_RESULTS / "metrics/filt/filtered_expert_rules_trained_predictions.csv")

cr_filter_nn = pd.read_csv(
    NN_RESULTS / "metrics/cr/cross_expert_rules_predictions.csv")

print(cr_filter_nn.shape)

# Check if the dfs are in the right shape
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
filter_nn.columns = drugs

print(drugs)

# Original order: 'Amikacin', 'Bedaquiline', 'Capreomycin', 'Clofazimine', 'Delamanid',
# 'Ethambutol', 'Ethionamide', 'Isoniazid', 'Kanamycin', 'Levofloxacin', 'Linezolid',
# 'Moxifloxacin', 'Pyrazinamide', 'Rifampicin', 'Streptomycin'

# The order that the cr preds are made are as follows:
# 'Amikacin', 'Capreomycin', 'Delamanid', 'Ethambutol', 'Kanamycin', 'Linezolid', 'Pyrazinamide',
# 'Rifampicin', 'Streptomycin','Levofloxacin','Moxifloxacin','Bedaquiline','Clofazimine',
# 'Isoniazid','Ethionamide'
# 11 (BDQ) -> 1, 12 (CFZ) -> 3, 14 (ETO) -> 6, 13 (INH) -> 7, 9 (LFX) -> 9, 10 (MFX) -> 11

new_order = [0, 11, 1, 12, 2, 3, 14, 13, 4, 9, 5, 10, 6, 7, 8]
# amk, bdq, cap, cfz, dlm, emb, eto, inh, kan, lfx, lzd, mfx, pza, rif, stp
cr_filter_nn = cr_filter_nn.iloc[:, new_order]
cr_filter_nn.columns = drugs
print(cr_filter_nn.shape)

# load neural network runs
runs = pd.read_csv(NN_RESULTS / 'essentials/vcf_list.txt',
                   sep=" ", header=None)
runs.columns = ['run']
runs['run'] = runs['run'].apply(lambda x: x.split("/")[-1].split(".")[0])
who_nn = pd.merge(runs, who_nn, left_index=True, right_index=True, how='inner')
er_nn = pd.merge(runs, er_nn, left_index=True, right_index=True, how='inner')
filter_nn = pd.merge(runs, filter_nn, left_index=True,
                     right_index=True, how='inner')
cr_filter_nn = pd.merge(runs, cr_filter_nn, left_index=True,
                        right_index=True, how='inner')

who_nn_class = pd.merge(who_nn, truth, on='run', how='inner')
er_nn_class = pd.merge(er_nn, truth, on='run', how='inner')
filter_nn_class = pd.merge(filter_nn, truth, on='run', how='inner')
cr_filter_nn_class = pd.merge(cr_filter_nn, truth, on='run', how='inner')

cl_dt = {}
for name, df in [("who_nn_class", who_nn_class), ("er_nn_class", er_nn_class), ("filter_nn_class", filter_nn_class), ("cr_filter_nn_class", cr_filter_nn_class)]:
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
filter_nn_flt = cl_dt["filter_nn_class"]
cr_filter_nn_flt = cl_dt["cr_filter_nn_class"]

results = []
for drug in lower_drugs:
    who_nn_subset = who_nn_flt[who_nn_flt['drug'] == drug]
    er_nn_subset = er_nn_flt[er_nn_flt['drug'] == drug]
    filter_nn_subset = filter_nn_flt[filter_nn_flt['drug'] == drug]
    cr_filter_nn_subset = cr_filter_nn_flt[cr_filter_nn_flt['drug'] == drug]
    tb_prof_subset = tb_prof[tb_prof['drug'] == drug]
    subsets = {'who_nn': who_nn_subset,
               'er_nn': er_nn_subset, 'filter_nn': filter_nn_subset, 'cr_filter_nn': cr_filter_nn_subset, 'tb': tb_prof_subset}
    cnts = defaultdict(dict)
    for cl in ['TN', 'TP', 'FN', 'FP']:
        for tool in ['who_nn', 'er_nn', 'filter_nn', 'cr_filter_nn', 'tb']:
            cnts[tool][cl] = int(subsets[tool]['classification'].eq(cl).sum())
    mets = defaultdict(dict)

    def safe_div(n, d):
        return n / d if d != 0 else float('nan')
    for tool in ['who_nn', 'er_nn', 'filter_nn', 'cr_filter_nn', 'tb']:
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

    def na_format(tb, who, er, filter_nn, cr_filter_nn):
        def fmt(x):
            return f"{x:.3f}" if not math.isnan(x) else "NA"
        return f"{fmt(tb)} / ({fmt(who)}, {fmt(er)}, {fmt(filter_nn)}, {fmt(cr_filter_nn)})"
    tb = mets['tb']
    who = mets['who_nn']
    er = mets['er_nn']
    filt = mets['filter_nn']
    cr_filt = mets['cr_filter_nn']
    results.append({
        "Drug": drug,
        "Sensitivity": na_format(tb['sensitivity'], who['sensitivity'], er['sensitivity'], filt['sensitivity'], cr_filt['sensitivity']),
        "Specificity": na_format(tb['specificity'], who['specificity'], er['specificity'], filt['specificity'], cr_filt['specificity']),
        "Precision": na_format(tb['precision'], who['precision'], er['precision'], filt['precision'], cr_filt['precision']),
        "F1": na_format(tb['f1'], who['f1'], er['f1'], filt['f1'], cr_filt['f1'])
    })
results_df = pd.DataFrame(results)
title = f'#Metrics in form tb-profiler / (WHO snps NN, expert rules NN, filtered expert rules NN, cr_filter_nn)\n'
with open(NN_RESULTS / "metrics/naiive_perf_metrics.csv", "w") as f:
    f.write(title)
    results_df.to_csv(f, index=False)
