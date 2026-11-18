"""
Write the run-ID list for each FN/FP/TN/TP matrix built by makematrix.py.

makematrix.py builds {classif}_{drug}_snps.npz with one row per accession,
in the order of run_dict[classif][drug] from NEW_updated_runs.json, but
doesn't save which accession each row is. enrich_snps.py (fischer.py) needs
that to keep only training isolates. This writes {classif}_{drug}_runs.txt
next to each matrix, one accession per line in row order, without re-reading
any VCFs.

Only valid if NEW_updated_runs.json hasn't changed since the matrices were
built; enrich_snps.py checks that each list's length matches its matrix.

    python scripts/write_run_sidecars.py
    python scripts/write_run_sidecars.py --old-tbprof /path/to/old_tbprof/results
"""

import argparse
import json
from pathlib import Path

import scipy.sparse as sp


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--old-tbprof', default=(
        '/data/scratch/projects/punim1703/false_negative/old_tbprof/results'))
    args = ap.parse_args()

    old = Path(args.old_tbprof)
    with open(old / 'essentials/NEW_updated_runs.json') as f:
        run_dict = json.load(f)

    n_written, problems = 0, []
    for classif in ['FN', 'FP', 'TN', 'TP']:
        for drug, runs in run_dict.get(classif, {}).items():
            npz = old / f'matrices/{drug}/{classif}_{drug}_snps.npz'
            if not runs or not npz.exists():
                continue
            n_rows = sp.load_npz(npz).shape[0]
            if n_rows != len(runs):
                problems.append(f"{npz.name}: {n_rows} rows but {len(runs)} runs in the JSON")
                continue
            npz.with_name(f'{classif}_{drug}_runs.txt').write_text('\n'.join(runs) + '\n')
            n_written += 1

    print(f"wrote {n_written} run lists")
    if problems:
        print("NOT written (JSON no longer matches the matrix; rebuild with makematrix.py):")
        print('\n'.join('  ' + p for p in problems))


if __name__ == '__main__':
    main()
