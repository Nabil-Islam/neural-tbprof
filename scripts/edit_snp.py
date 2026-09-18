import json
from pathlib import Path
import re
import subprocess
import pandas as pd
from cyvcf2 import VCF
from tqdm import tqdm

RESULTS = Path.cwd() / "results"
O_RESULTS = Path(
    "/data/scratch/projects/punim1703/false_negative/old_tbprof/results")
MATRICES = O_RESULTS / "matrices"
results = subprocess.run(
    f"ls {MATRICES}", shell=True, capture_output=True, text=True)
drugs = [line.strip("/") for line in results.stdout.split()]

gff = pd.read_csv(
    "/data/scratch/projects/punim1703/tbvcf/resources/NC_000962.3.gff3",
    sep="\t",
    comment="#",
    header=None,
    names=[
        "seqid",
        "source",
        "type",
        "start",
        "end",
        "score",
        "strand",
        "phase",
        "attributes",
    ],
)

gff = gff[gff["type"].isin(["gene", "CDS"])].copy()

gff["locus_tag"] = gff["attributes"].str.extract(r"(Rv\d{4}c?)").fillna("null")
gff["name"] = gff["attributes"].str.extract(r";gene=([^;]+)").fillna("null")

gff["start"] = gff["start"].astype(int)
gff["end"] = gff["end"].astype(int)

gff_intervals = pd.IntervalIndex.from_arrays(
    gff["start"], gff["end"], closed="both"
)

output = {}

for drug in drugs:
    csv_path = MATRICES / drug / f"filtered_{drug}_snps.csv"

    try:
        df = pd.read_csv(csv_path)
    except (pd.errors.EmptyDataError, FileNotFoundError, NotADirectoryError):
        continue

    if "snp" not in df.columns or df.empty:
        continue

    # Pre-calculate positions once per drug CSV to save processing time
    df["pos_int"] = df["snp"].astype(str).str.split("_").str[0].astype(int)

    drug_genes = set()

    for row in df.itertuples(index=False):
        pos = row.pos_int
        snp = row.snp
        fn_snp_prop = (
            round(float(row.fn_snp_prop), 2)
            if pd.notna(row.fn_snp_prop)
            else 0.0
        )
        fn_no_snp_prop = (
            round(float(row.fn_no_snp_prop), 2)
            if pd.notna(row.fn_no_snp_prop)
            else 0.0
        )

        # Find which row(s) in the GFF contain this position
        matching_indices = gff_intervals.contains(pos)
        matched_gff = gff[matching_indices]

        # Iterate over matching GFF elements
        for gff_row in matched_gff.itertuples(index=False):
            name = gff_row.name
            locus_tag = gff_row.locus_tag

            if name == "null" or name == "material":
                name = locus_tag if locus_tag != "null" else "intergenic region"

            print(drug, name, pos, fn_snp_prop, fn_no_snp_prop)

            entry = (drug, name, snp, fn_snp_prop, fn_no_snp_prop)
            drug_genes.add(entry)

    if drug_genes:
        output[drug] = [
            {
                "drug": item[0],
                "gene_or_locus": item[1],
                "snp": item[2],
                "fn_snp_prop": item[3],
                "fn_no_snp_prop": item[4],
            }
            for item in drug_genes
        ]

with open(RESULTS / "filtered_gene_name.json", "w") as f:
    json.dump(output, f, indent=4)
