"""
Variant enrichment in TB-Profiler false negatives, on TRAINING isolates only.

For each drug, compares variant frequency between
    FN : phenotypically resistant, TB-Profiler called susceptible
    TN : phenotypically susceptible, TB-Profiler called susceptible
with a Fisher exact test, and writes results/matrices/<drug>/<drug>_enriched_snps.csv
(same columns as before, so flag_snps.py -> filtered_snps.py work unchanged).

This uses phenotype labels, so it is part of model fitting: test isolates
must not take part, or the `filt` variant set is chosen partly by the test
set. Rows are restricted using results/essentials/splits.csv (make_splits.py).

Each FN_<drug>_snps.npz / TN_<drug>_snps.npz must have a sidecar file
FN_<drug>_runs.txt / TN_<drug>_runs.txt listing the accession of each
matrix row, one per line, in row order (save it where the matrices are built).

usage (from repo root):
    python scripts/enrich_snps.py
    python scripts/enrich_snps.py --exclude-fold 0   # also keep train_new.py's
                                                     # validation fold out
"""

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.stats import fisher_exact
from tqdm import tqdm


def load_rows(npz_path, splits, keep_folds):
    """Load an FN/TN matrix and keep only rows whose accession is in keep_folds."""
    runs_path = npz_path.with_name(npz_path.name.replace('_snps.npz', '_runs.txt'))
    if not runs_path.exists():
        raise SystemExit(
            f"missing {runs_path}: save the accession of each row of "
            f"{npz_path.name} (in row order) where that matrix is built")

    runs = pd.Index([r.strip() for r in runs_path.read_text().split('\n') if r.strip()])
    m = sp.load_npz(npz_path).tocsr()
    if m.shape[0] != len(runs):
        raise SystemExit(f"{npz_path.name}: {m.shape[0]} rows but {len(runs)} run IDs")

    missing = runs.difference(splits.index)
    if len(missing):
        raise SystemExit(f"{len(missing)} accessions in {runs_path.name} are not "
                         f"in splits.csv (e.g. {list(missing[:3])})")

    s = splits.loc[runs]
    keep = ((s['split'] == 'train') & s['fold'].isin(keep_folds)).to_numpy()
    return m[keep].tocsc(), int(keep.sum()), len(runs)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--matrices', default='results/matrices')
    ap.add_argument('--snps', default='results/essentials/NEW_snps.txt')
    ap.add_argument('--splits', default='results/essentials/splits.csv')
    ap.add_argument('--exclude-fold', type=int, action='append', default=[],
                    help='CV fold(s) to leave out as well (repeatable)')
    ap.add_argument('--alpha', type=float, default=0.05,
                    help='p-value cut-off for writing a variant (as before)')
    args = ap.parse_args()

    matrices = Path(args.matrices)
    snps = np.array([s.strip() for s in Path(args.snps).read_text().split('\n') if s.strip()])

    splits = pd.read_csv(args.splits, dtype={'run': str}).set_index('run')
    all_folds = sorted(int(f) for f in splits.loc[splits['split'] == 'train', 'fold'].unique())
    keep_folds = [f for f in all_folds if f not in set(args.exclude_fold)]
    print(f"using training folds {keep_folds} (test set and folds "
          f"{args.exclude_fold or 'none'} excluded)")

    drugs = sorted(p.name for p in matrices.iterdir() if p.is_dir())
    log = []

    for drug in tqdm(drugs, desc='drugs'):
        fn_path = matrices / drug / f'FN_{drug}_snps.npz'
        tn_path = matrices / drug / f'TN_{drug}_snps.npz'
        if not fn_path.exists() or not tn_path.exists():
            continue

        fn, n_fn, n_fn_all = load_rows(fn_path, splits, keep_folds)
        tn, n_tn, n_tn_all = load_rows(tn_path, splits, keep_folds)
        log.append({'drug': drug, 'fn_used': n_fn, 'fn_total': n_fn_all,
                    'tn_used': n_tn, 'tn_total': n_tn_all})

        if fn.shape[1] != len(snps) or tn.shape[1] != len(snps):
            raise SystemExit(f"{drug}: matrix has {fn.shape[1]}/{tn.shape[1]} "
                             f"columns but {len(snps)} SNP names")

        fn_ones = np.asarray(fn.sum(axis=0)).ravel()
        tn_ones = np.asarray(tn.sum(axis=0)).ravel()
        fn_zeros = n_fn - fn_ones
        tn_zeros = n_tn - tn_ones

        with np.errstate(invalid='ignore', divide='ignore'):
            fn_snp_prop = fn_ones / (fn_ones + tn_ones)
            fn_no_snp_prop = fn_zeros / (fn_zeros + tn_zeros)

        results = []
        # variants absent from both groups can't be enriched; skip the test
        for i in np.flatnonzero(fn_ones + tn_ones > 0):
            res = fisher_exact([[fn_ones[i], fn_zeros[i]],
                                [tn_ones[i], tn_zeros[i]]])
            p, odds = res.pvalue, res.statistic
            if p <= args.alpha:
                results.append({
                    'snp': snps[i],
                    'log_pval': round(math.log10(p), 3) if p > 0 else -np.inf,
                    'log_or': round(math.log10(odds), 3) if odds > 0 else -np.inf,
                    'fn_snp_prop': fn_snp_prop[i],
                    'fn_no_snp_prop': fn_no_snp_prop[i],
                    'fn_count': int(fn_ones[i]),
                })

        df = pd.DataFrame(results, columns=['snp', 'log_pval', 'log_or', 'fn_snp_prop',
                                            'fn_no_snp_prop', 'fn_count'])
        if df.empty:
            print(f"no enriched variants for {drug}")
        df.sort_values('log_pval', ascending=False).to_csv(
            matrices / drug / f'{drug}_enriched_snps.csv', index=False)

    pd.DataFrame(log).to_csv(matrices / 'enrichment_rows_used.csv', index=False)
    print(pd.DataFrame(log).to_string(index=False))


if __name__ == '__main__':
    main()
