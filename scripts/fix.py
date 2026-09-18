import json
import re
from pathlib import Path
import pandas as pd

DIFF_SNPS_JSON = Path.cwd() / "diff_snps.json"
FIXED_SNPS_JSON = Path.cwd() / "diff_snps_fixed.json"
GFF_PATH = Path(
    "/data/scratch/projects/punim1703/tbvcf/resources/NC_000962.3.gff3")
EXTRA = Path.cwd().parent / "filtered_gene_name.json"
FLAGGED = FLAGGED = Path(
    '/data/scratch/projects/punim1703/false_negative/old_tbprof/results')

gff = pd.read_csv(
    GFF_PATH,
    sep="\t",
    comment="#",
    header=None,
    names=["seqid", "source", "type", "start", "end",
           "score", "strand", "phase", "attributes"]
)

# Only keep actual genes/CDS features to avoid "material" artifacts
gff = gff[gff["type"].isin(["gene", "CDS"])].copy()
gff["locus_tag"] = gff["attributes"].str.extract(r"(Rv\d{4}c?)").fillna("null")

gff["start"] = gff["start"].astype(int)
gff["end"] = gff["end"].astype(int)

gff_intervals = pd.IntervalIndex.from_arrays(
    gff['start'], gff['end'], closed='both')

with open(EXTRA, "r") as f:
    extra_raw = json.load(f)

fn_lookup = {}
for drug, entries in extra_raw.items():
    for entry in entries:
        snp_name = entry.get("snp")
        fn_prop = entry.get("fn_snp_prop")
        if snp_name and fn_prop is not None:
            fn_lookup[(drug, snp_name)] = fn_prop

with open(DIFF_SNPS_JSON, "r") as f:
    data = json.load(f)

flagged_snps = pd.read_csv(FLAGGED/'flagged_snps.csv')['snp'].tolist()
flagged_snps_set = set(flagged_snps)

fixed_count = 0
added_props_count = 0
rem_count = 0

for drug, snps in data.items():
    for snp_key, values_list in snps.items():

        keys_to_remove = [
            snp_key for snp_key in snps if snp_key in flagged_snps_set]

        if len(values_list) > 1 and values_list[1] in ["gene not found", "material"]:
            pos = int(snp_key.split('_')[0])
            matching_indices = gff_intervals.contains(pos)
            matched_gff = gff[matching_indices]

            if not matched_gff.empty:
                valid_tags = matched_gff[matched_gff['locus_tag']
                                         != 'null']['locus_tag']
                if not valid_tags.empty:
                    values_list[1] = valid_tags.iloc[0]
                    fixed_count += 1
                else:
                    values_list[1] = "intergenic region"
            else:
                values_list[1] = "intergenic region"

        fn_prop = fn_lookup.get((drug, snp_key))

        if fn_prop is not None:
            # Append only if it's not already tracked in the
            if fn_prop not in values_list:
                values_list.append(f"{fn_prop}%")
                added_props_count += 1

    for snp_key in list(snps.keys()):
        if snp_key in flagged_snps:
            del snps[snp_key]
            rem_count += 1

print(f"removed {rem_count} from flagged_snps")

with open(FIXED_SNPS_JSON, "w") as f:
    json.dump(data, f, indent=4)
