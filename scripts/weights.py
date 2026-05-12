from pathlib import Path
import json
import numpy as np
import scipy.sparse as sp
import sys

workflow = Path.cwd()
results = workflow/"results"
arg = sys.argv[1]

if arg == "er":
    with open(results/'essentials/combined_snps.json', 'r') as f:
        weight_m = json.load(f)
    name = "expert_rules"

if arg == "filt":
    with open(results/'essentials/filtered_combined_snps.json', 'r') as f:
        weight_m = json.load(f)
    name = "filtered_expert_rules"

if arg == "who":
    with open(results/'essentials/who_snps.json', 'r') as f:
        weight_m = json.load(f)
    name = "who"

with open(results/f"matrices/{name}/{name}_LOOKUP_SNP.json", 'r') as f:
    snp_lu = json.load(f)

print(name)
drugs = list(weight_m.keys())
drug_lu = {drug: i for i, drug in enumerate(drugs)}
print(drug_lu)

rows = []
cols = []

for drug, snps in weight_m.items():
    row_idx = drug_lu[drug]
    for snp in snps:
        col_idx = snp_lu.get(snp)
        if col_idx is not None:
            rows.append(row_idx)
            cols.append(col_idx)

rows = np.array(rows, dtype=np.int32)
cols = np.array(cols, dtype=np.int32)
data = np.ones(len(rows), dtype=np.int8)

matrix = sp.coo_matrix(
    (data, (rows, cols)),
    shape=(len(drugs), len(snp_lu.keys()))
)

sp.save_npz(results/f"matrices/{name}/{name}_WEIGHTS_AUG.npz", matrix)
