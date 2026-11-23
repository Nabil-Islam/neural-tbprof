import pandas as pd
from cyvcf2 import VCF
import re
import json
from tqdm import tqdm

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

# WHO catalogue 2nd ed. (2023), Table 1 "additional grading rules" (-> Group 2):
#   rpoB RRDR (codons 426-452): any non-silent change         -> RIF
#   all other rules are loss-of-function only (in-frame indels excluded):
#   katG (INH), pncA (PZA), Rv0678/pepQ (BDQ, CFZ),
#   ddn/fbiA/fbiB/fbiC/fgd1/Rv2983 (DLM), gid (STM), ethA (ETO), tlyA (CAP)
# An earlier version treated pncA, ethA, tlyA, Rv0678/pepQ and the DLM genes
# as "any non-synonymous change", which pulled in many neutral missense
# polymorphisms and cut sensitivity at matched specificity.
ALL_NONSYN_GENES = {}

RRDR_CODONS = range(426, 453)  # rpoB RRDR

LOF_TYPES = {'stop_gained', 'frameshift_variant',
             'feature_ablation', 'start_lost'}

LOF_ONLY_GENES = {
    'isoniazid':    ['katG'],
    'streptomycin': ['gid'],
    'pyrazinamide': ['pncA'],
    'ethionamide':  ['ethA'],
    'capreomycin':  ['tlyA'],
    'bedaquiline':  ['Rv0678', 'pepQ'],
    'clofazimine':  ['Rv0678', 'pepQ'],
    'delamanid':    ['ddn', 'fbiA', 'fbiB', 'fbiC', 'fgd1', 'Rv2983'],
}

snp_dict = {drug: {} for drug in drug_dict.keys()}
print(snp_dict)

with open('/data/scratch/projects/punim1703/nn_tb/REVISION/results/essentials/vcf_list.txt', "r") as f:
    vcfs = f.readlines()

vcfs = [vcf.strip('\n') for vcf in vcfs]

# Expert rules are applied to every isolate's variants, regardless of
# phenotype, the way TB-Profiler applies them. (An earlier version only
# looked at isolates resistant to each drug, across the whole dataset: that
# used test-set phenotypes to choose features.)


def codon_from_hgvs_p(hgvs_p):
    """'p.Ser450Leu' -> 450. HGVS.p (ANN field 10) holds the amino-acid change;
    HGVS.c (field 9) holds a nucleotide position, which is not a codon."""
    m = re.match(r'p\.[A-Za-z*]+(\d+)', hgvs_p)
    return int(m.group(1)) if m else None


def rule_applies(drug_l, gene_name, effect, hgvs_p):
    """True if an expert rule makes this variant a resistance feature for drug."""
    if effect == 'synonymous_variant':
        return False
    if drug_l == 'rifampicin' and gene_name == 'rpoB':        # RRDR rule
        codon = codon_from_hgvs_p(hgvs_p)
        return codon is not None and codon in RRDR_CODONS
    if gene_name in ALL_NONSYN_GENES.get(drug_l, []):          # any non-synonymous change
        return True
    if gene_name in LOF_ONLY_GENES.get(drug_l, []):            # loss-of-function only
        return effect in LOF_TYPES
    return False                                               # no rule for this drug/gene


for vcf in tqdm(vcfs, total=len(vcfs)):
    for variant in VCF(vcf):
        ann = variant.INFO.get('ANN')
        if not ann:
            continue
        pos = variant.start + 1
        ref = variant.REF

        # one variant can carry several annotations (overlapping genes,
        # upstream regions, other alleles): check each, not just the first
        for entry in ann.split(','):
            f = entry.split('|')
            if len(f) < 11:
                continue
            allele, effects, gene_id, hgvs_p = f[0], f[1], f[3], f[10]
            if allele not in variant.ALT:
                continue
            gene_name = locus_to_name.get(gene_id, gene_id)

            # an annotation can list several effects joined by '&'
            for effect in effects.split('&'):
                for drug, genes in drug_dict.items():
                    if gene_name not in genes:
                        continue
                    if rule_applies(drug.lower(), gene_name, effect, hgvs_p):
                        snp_dict.setdefault(drug, {})[f'{pos}_{ref}_{allele}'] = 1

print(drug_dict)
print(snp_dict)

with open("/data/scratch/projects/punim1703/nn_tb/REVISION/results/essentials/snp_aug.json", "w") as f:
    json.dump(snp_dict, f)
