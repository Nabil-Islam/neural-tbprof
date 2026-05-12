import pandas as pd
import json
from pathlib import Path

snps_path = Path(
    "/data/scratch/projects/punim1703/false_negative/old_tbprof/results/matrices")
nn_path = Path("/data/scratch/projects/punim1703/nn_tb/REVISION/")
with open(nn_path/"results/essentials/FINAL_SNP.json", "r") as f:
    snp_json = json.load(f)
drugs = list(snp_json.keys())

enriched_snps = {d: '' for d in drugs}

for drug in drugs:
    drug_file = pd.read_csv(
        snps_path / f"{drug.lower()}/{drug.lower()}_enriched_snps.csv")
    subset = drug_file[drug_file['fn_count'] >= 470]
    snps = subset['snp']
    enriched_snps[drug] = list(snps)

with open(nn_path/"results/essentials/enriched_snps.json", "w") as f:
    json.dump(enriched_snps, f)
