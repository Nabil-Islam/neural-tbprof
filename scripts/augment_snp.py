import pandas as pd
from cyvcf2 import VCF
import re
import json
from tqdm import tqdm

samples = pd.read_csv(
    '/data/scratch/projects/punim1703/false_negative/old_tbprof/resources/data/samples.csv')

who_df = pd.read_csv(
    '/data/scratch/projects/punim1703/nn_tb/REVISION/data/sheet1.csv', skiprows=2, low_memory=False)

gff = pd.read_csv("/data/scratch/projects/punim1703/tbvcf/resources/NC_000962.3.gff3",
                  sep="\t", comment="#", header=None)
gff.columns = ["seqid", "source", "type", "start",
               "end", "score", "strand", "phase", "attributes"]

gff["locus_tag"] = 'null'

gff["name"] = 'null'

for index, row in gff.iterrows():
    id = re.search(r"ID=[^-]+-(\w+)", row["attributes"])
    name = re.search(r";Name=([^;]+)", row["attributes"])

    gff.loc[index, 'locus_tag'] = id.group(1) if id else 'null'
    gff.loc[index, 'name'] = name.group(1) if name else 'null'

for index, row in who_df.iterrows():
    who_df.at[index, "FINAL CONFIDENCE GRADING"] = row["FINAL CONFIDENCE GRADING"][0]

who_df = who_df.loc[
    (who_df["FINAL CONFIDENCE GRADING"] == "1")
    | (who_df["FINAL CONFIDENCE GRADING"] == "2")
]

drug_dict = {}

for i, row in who_df.iterrows():
    drug = who_df.loc[i, 'drug'].lower()
    gene = who_df.loc[i, 'gene']

    if drug not in drug_dict:
        drug_dict[drug] = []
    if gene not in drug_dict[drug]:
        drug_dict[drug].append(gene)

for drug, genes in drug_dict.items():
    for i, gene in enumerate(genes):
        match = gff.loc[gff['name'] == gene, 'locus_tag']
        drug_dict[drug][i] = match.iloc[0] if not match.empty else None

print(drug_dict)

INCLUDE_DRUGS = ('rifampicin', 'isoniazid', 'pyrazinamide',
                 'ethionamide', 'streptomycin')

snp_dict = {drug: {} for drug in drug_dict.keys()}
print(snp_dict)

with open('/data/scratch/projects/punim1703/nn_tb/REVISION/results/vcf_list.txt', "r") as f:
    vcfs = f.readlines()

vcfs = [vcf.strip('\n') for vcf in vcfs]

for vcf in tqdm(vcfs, total=len(vcfs)):

    run = vcf.split("/")[-1].split(".")[0]

    sample_row = samples.loc[samples['run'] == run]

    if sample_row.empty:
        continue

    sample_row = sample_row.iloc[0]

    resistant_drugs = {
        drug: genes
        for drug, genes in drug_dict.items()
        if sample_row.get(drug) == 'R'
    }

    if not resistant_drugs:
        continue

    for variant in VCF(vcf):
        ann = variant.INFO.get('ANN')

        if not ann:
            continue

        ann_s = ann.split('|')

        var_type = ann_s[1]

        # Any non-expert rule drugs should be removed remove Lefo and Moxi

        if var_type in ('synonymous_variant'):
            continue

        gene_name = ann_s[3]

        for drug, genes in resistant_drugs.items():
            if drug.lower() not in INCLUDE_DRUGS:
                continue
            if gene_name in genes:
                pos = variant.start + 1
                ref = variant.REF

                for alt in variant.ALT:
                    snp_str = f'{pos}_{ref}_{alt}'
                    snp_dict.setdefault(drug, {})[snp_str] = 1
print(snp_dict)

with open("/data/scratch/projects/punim1703/nn_tb/REVISION/results/snp_aug.json", "w") as f:
    json.dump(snp_dict, f)
