from pathlib import Path
from cyvcf2 import VCF
import json
from tqdm import tqdm
import scipy.sparse as sp
import numpy as np
import sys
import filtered_snps

workflow = Path.cwd()
results = workflow/"results"
data = workflow/"data"
arg = sys.argv[1]

with open(results / "essentials/who_snps.json") as f:
    weight_m = json.load(f)
name = "who"

if arg == "er":
    name = "expert_rules"
    with open(results / 'essentials/snp_aug.json') as f:
        new_snp = json.load(f)

    for drug, snps in new_snp.items():
        for snp in snps:
            if snp in weight_m.get(drug.capitalize(), {}):
                continue
            weight_m.setdefault(drug.capitalize(), {})[snp] = 1

    with open(results / 'essentials/combined_snps.json', 'w') as f:
        json.dump(weight_m, f)

# Generate combined filtered snps depending logpval and log or
comb_snps = Path(results / "essentials/combined_snps.json").exists()
if arg == "filt" and comb_snps:
    # filtered_snps.main()
    with open(results / 'essentials/filtered_combined_snps.json', 'r') as f:
        weight_m = json.load(f)
    name = "filtered_expert_rules"

snp_set = set()

for drug, snps in weight_m.items():
    for snp in snps:
        snp_set.add(snp)

snp_list = list(snp_set)

snp_lu = {snp: i for i, snp in enumerate(snp_list)}

with open(results/f"matrices/{name}/{name}_LOOKUP_SNP.json", 'w') as f:
    json.dump(snp_lu, f)

with open(results/"essentials/vcf_list.txt") as f:
    vcfs = f.readlines()

vcfs = [vcf.strip('\n') for vcf in vcfs]

rows = []
cols = []

for i, vcf_path in enumerate(tqdm(vcfs, desc=f"{name} making matrix")):
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

sp.save_npz(results/f"matrices/{name}/{name}_AUG_matrix.npz", matrix)
