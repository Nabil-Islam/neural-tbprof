from cyvcf2 import VCF
import pandas as pd
from pathlib import Path
import json

# --- Need to construct a 15 drug long vector for each vcf ---
# --- 1 for resistant, 0 for susceptible ---
# --- Impute missing values with 1 ---

# PATHS
SCRIPTS = Path.cwd()
RESULTS = Path(SCRIPTS / 'results')
OLD_TB = Path(
    "/data/scratch/projects/punim1703/false_negative/old_tbprof/resources/data")

# STORE DRUG LIST
with open(RESULTS / "FINAL_SNP.json") as f:
    drug_snps = json.load(f)

drugs = list(drug_snps.keys())
lower_drugs = [d.lower() for d in drugs]

# STORE GROUND TRUTHS
ground_truth = pd.read_csv(OLD_TB / "samples.csv")

# ALL VCFS TO BE CHECKED
runs = pd.read_csv(RESULTS / 'vcf_list.txt', sep=" ", header=None)
runs.columns = ['run']
runs['run'] = runs['run'].apply(lambda x: x.split("/")[-1].split(".")[0])

# MERGE VCFS WITH RUNS
labelled_vcfs = pd.DataFrame(None, index=runs['run'], columns=lower_drugs)
for _, row in ground_truth.iterrows():
    for drug in lower_drugs:
        if pd.notna(row[drug]):
            if row[drug] == 'S':
                labelled_vcfs.loc[row['run'], drug] = 0
            elif row[drug] == 'R':
                labelled_vcfs.loc[row['run'], drug] = 1
labelled_vcfs.to_csv(RESULTS / 'labelled_vcfs.csv')
