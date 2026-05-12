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

locus_to_name = {}
for _, row in gff.iterrows():
    if row['locus_tag'] != 'null' and row['name'] != 'null':
        locus_to_name[row['locus_tag']] = row['name']

for name in ['katG', 'pncA', 'rpoB', 'gyrA']:
    match = {k: v for k, v in locus_to_name.items() if v == name}
    print(f"{name}: {match}")

print(locus_to_name.get('Rv0678'))

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

print(drug_dict)

ALL_NONSYN_GENES = {
    'pyrazinamide': ['pncA'],
    'ethionamide':  ['ethA'],
    'capreomycin':  ['tlyA'],
    'bedaquiline':  ['Rv0678', 'pepQ'],
    'clofazimine':  ['Rv0678', 'pepQ'],
    'delamanid':    ['ddn', 'fbiA', 'fbiB', 'fbiC', 'fgd1', 'Rv2983'],
}

RRDR_CODONS = range(426, 453)  # rpoB RRDR

LOF_TYPES = {'stop_gained', 'frameshift_variant',
             'feature_ablation', 'start_lost'}

LOF_ONLY_GENES = {
    'isoniazid': ['katG'],
    'streptomycin': ['gid']
}

snp_dict = {drug: {} for drug in drug_dict.keys()}
print(snp_dict)

with open('/data/scratch/projects/punim1703/nn_tb/REVISION/results/essentials/vcf_list.txt', "r") as f:
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

        if var_type in ('synonymous_variant'):
            continue

        gene_name = locus_to_name.get(ann_s[3], ann_s[3])
        for drug, genes in resistant_drugs.items():

            drug_l = drug.lower()
            if gene_name not in genes:
                continue

            # rpoB RRDR rule - non-synonymous with RRDR only
            if drug_l == 'rifampicin' and gene_name == 'rpoB':
                # adjust parsing as needed
                codon_match = re.search(r'\d+', ann_s[9])
                if not codon_match:
                    continue
                codon = int(codon_match.group())
                if var_type == 'synonymous_variant' or codon not in RRDR_CODONS:
                    continue

            # LoF-only genes
            elif gene_name in ALL_NONSYN_GENES.get(drug_l, []):
                pass

            elif gene_name in LOF_ONLY_GENES.get(drug_l, []):
                if var_type not in LOF_TYPES:
                    continue

            # No expert rule for this drug/gene combo - just old method, where we picked up LoF variants
            else:
                continue

            pos = variant.start + 1
            ref = variant.REF

            for alt in variant.ALT:
                snp_str = f'{pos}_{ref}_{alt}'
                snp_dict.setdefault(drug, {})[snp_str] = 1
print(drug_dict)
print(snp_dict)

with open("/data/scratch/projects/punim1703/nn_tb/REVISION/results/essentials/snp_aug.json", "w") as f:
    json.dump(snp_dict, f)
