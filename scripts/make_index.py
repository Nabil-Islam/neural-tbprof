from pathlib import Path
from cyvcf2 import VCF
import json
from tqdm import tqdm
import scipy.sparse as sp
import numpy as np
import sys

workflow = Path.cwd()
results = workflow/"results"
data = workflow/"data"
arg = sys.argv[1]

with open("/data/scratch/projects/punim1703/nn_tb/REVISION/results/who_snps.json") as f:
    weight_m = json.load(f)
name = "who"

if arg == "er":
    name = "expert_rules"
    with open('/data/scratch/projects/punim1703/nn_tb/REVISION/results/snp_aug.json') as f:
        new_snp = json.load(f)

    for drug, snps in new_snp.items():
        for snp in snps:
            if snp in weight_m.get(drug.capitalize(), {}):
                continue
            weight_m.setdefault(drug.capitalize(), {})[snp] = 1

    with open('/data/scratch/projects/punim1703/nn_tb/REVISION/results/combined_snps.json', 'w') as f:
        json.dump(weight_m, f)

snp_set = set()

for drug, snps in weight_m.items():
    for snp in snps:
        snp_set.add(snp)

snp_list = list(snp_set)

snp_lu = {snp: i for i, snp in enumerate(snp_list)}

with open(results/f"{name}_LOOKUP_SNP.json", 'w') as f:
    json.dump(snp_lu, f)

with open(results/"vcf_list.txt") as f:
    vcfs = f.readlines()

vcfs = [vcf.strip('\n') for vcf in vcfs]

rows = []
cols = []

for i, vcf_path in enumerate(tqdm(vcfs, desc="making matrix")):
    vcf = VCF(vcf_path)
    for variant in vcf:
        pos = variant.start + 1
        ref = variant.REF

        for alt in variant.ALT:
            snp_str = f"{pos}_{ref}_{alt}"
            if snp_str in snp_lu.keys():
                rows.append(i)
                cols.append(snp_lu[snp_str])

rows = np.array(rows, dtype=np.int32)
cols = np.array(cols, dtype=np.int32)
data = np.ones(len(rows), dtype=np.int8)

matrix = sp.coo_matrix(
    (data, (rows, cols)),
    shape=(len(vcfs), len(snp_lu.keys()))
)

sp.save_npz(results/f"{name}_AUG_matrix.npz", matrix)
