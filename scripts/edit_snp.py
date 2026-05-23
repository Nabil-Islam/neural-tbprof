import pandas as pd
from cyvcf2 import VCF
import re
import json
from tqdm import tqdm
from pathlib import Path

RESULTS = Path.cwd()/'results/essentials'
with open(RESULTS/'sens_diff_snps.json', 'r') as f:
    good_snps = json.load(f)

gff = pd.read_csv("/data/scratch/projects/punim1703/tbvcf/resources/NC_000962.3.gff3",
                  sep="\t", comment="#", header=None)
gff.columns = ["seqid", "source", "type", "start",
               "end", "score", "strand", "phase", "attributes"]

gff["locus_tag"] = 'null'

gff["name"] = 'null'

for index, row in gff.iterrows():
    id = re.search(r"ID=[^-]+-(\w+)", row["attributes"])
    name = re.search(r";gene=([^;]+)", row["attributes"])

    gff.loc[index, 'locus_tag'] = id.group(1) if id else 'null'
    gff.loc[index, 'name'] = name.group(1) if name else 'null'

locus_to_name = {}
for _, row in gff.iterrows():
    if row['locus_tag'] != 'null' and row['name'] != 'null':
        locus_to_name[row['locus_tag']] = row['name']


for drug, snps in good_snps.items():
    for snp in snps:
        change = f'{round(good_snps[drug][snp]*100, 2)}%'
        pos = int(snp.split('_')[0])
        for index, row in gff.iterrows():
            if pos >= int(row['start']) and pos <= int(row['end']):
                print(row)
                name = row['name']
                if name == 'null':
                    name = 'gene not found'
                good_snps[drug][snp] = [change, name, f'{
                    snp.split('_')[1]}>{snp.split('_')[2]}']

with open(RESULTS/'diff_snps.json', 'w') as f:
    json.dump(good_snps, f)
