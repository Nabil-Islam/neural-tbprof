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

drugs = list(weight_m.keys())
# This order is how the cross r drugs will be stored
cr_drugs = [
    'Levofloxacin',
    'Moxifloxacin',
    'Bedaquiline',
    'Clofazimine',
    'Isoniazid',
    'Ethionamide'
]
for d in cr_drugs:
    if d in drugs:
        drugs.remove(d)
print(drugs)
print(cr_drugs)
drug_lu = {drug: i for i, drug in enumerate(drugs)}
cr_drug_lu = {d: i for i, d in enumerate(cr_drugs)}

print(drug_lu)
print(cr_drug_lu)

rows_r, cols_r, rows_cr, cols_cr, rows_cr_one, cols_cr_one = [], [], [], [], [], []

for drug in drugs:
    snps = weight_m.get(drug)
    row_idx = drug_lu[drug]
    for snp in snps:
        col_idx = snp_lu.get(snp)
        if col_idx is not None:
            rows_r.append(row_idx)
            cols_r.append(col_idx)

for drug in cr_drugs:
    snps = weight_m.get(drug)
    row_idx = cr_drug_lu.get(drug)
    for snp in snps:
        col_idx = snp_lu.get(snp)
        if col_idx is not None:
            rows_cr.append(row_idx)
            cols_cr.append(col_idx)
            rows_cr_one.append(row_idx)
            cols_cr_one.append(col_idx)
    rows_cr_one.append(row_idx)
    cols_cr_one.append(len(snp_lu) + row_idx)


print(cols_cr_one)

rows_r = np.array(rows_r, dtype=np.int32)
rows_cr_f = np.array(rows_cr, dtype=np.int32)
cols_r = np.array(cols_r, dtype=np.int32)
cols_cr_f = np.array(cols_cr, dtype=np.int32)

data_r = np.ones(len(rows_r), dtype=np.int8)
data_cr = np.ones(len(rows_cr), dtype=np.int8)

r_matrix = sp.coo_matrix(
    (data_r, (rows_r, cols_r)),
    shape=(len(drugs), len(snp_lu.keys()))
)
sp.save_npz(results/f"matrices/{name}/{name}_WEIGHTS_AUG.npz", r_matrix)

cr_matrix = sp.coo_matrix(
    (data_cr, (rows_cr_f, cols_cr_f)),
    shape=(len(cr_drugs), len(snp_lu.keys()))
)
sp.save_npz(results/f"matrices/{name}/{name}_CR_WEIGHTS_AUG.npz", cr_matrix)

rows_cr_one = np.array(rows_cr_one, dtype=np.int32)
cols_cr_one = np.array(cols_cr_one, dtype=np.int32)

data_cr_one = np.ones(len(rows_cr_one), dtype=np.int8)

cr_one_matrix = sp.coo_matrix(
    (data_cr_one, (rows_cr_one, cols_cr_one)),
    shape=(len(cr_drugs), len(snp_lu) + len(cr_drugs))
)
sp.save_npz(
    results/f"matrices/{name}/{name}_CR+1_WEIGHTS_AUG.npz", cr_one_matrix)
