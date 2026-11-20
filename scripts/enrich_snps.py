"""
Variant enrichment in TB-Profiler false negatives, on TRAINING isolates
only, stratified by sub-lineage.

For each drug, compares variant frequency between
    FN : phenotypically resistant, TB-Profiler called susceptible
    TN : phenotypically susceptible, TB-Profiler called susceptible
and writes results/matrices/<drug>/<drug>_enriched_snps.csv with the same
columns as before (snp, log_pval, log_or, fn_snp_prop, fn_no_snp_prop,
fn_count), so flag_snps.py -> filtered_snps.py work unchanged, plus:
    n_fn_strata   number of sub-lineages in which FN isolates carry the variant
    crude_log_or  unstratified log10 odds ratio, for comparison

Two problems this addresses:

1. Test-set leakage. Uses phenotype labels, so test isolates must not take
   part. Rows are restricted with results/essentials/splits.csv
   (make_splits.py); --exclude-fold also leaves out train_new.py's
   validation fold. Each FN_/TN_<drug>_snps.npz needs its
   FN_/TN_<drug>_runs.txt row list (write_run_sidecars.py).

2. Population structure. TB isolates aren't independent: if one lineage is
   over-represented among false negatives, all of its lineage-specific
   variants look "enriched". With --test cmh (default), the test is a
   Cochran-Mantel-Haenszel test stratified by sub-lineage (log_pval) with
   the Mantel-Haenszel common odds ratio (log_or): a variant has to be
   enriched in FNs *within* sub-lineages.
   That removes lineage markers, but not variants private to one outbreak
   cluster (still enriched within its own sub-lineage). n_fn_strata flags
   those: a resistance mutation tends to arise independently in several
   lineages; a cluster's private variants sit in one. --min-fn-strata 2
   drops variants seen in FNs of only one sub-lineage.
   --test fisher reproduces the old unstratified Fisher test.

Sub-lineages come from the `lineage` column of resources/data/samples.csv
(e.g. lineage2.2.1). Each isolate is placed at the deepest level, up to
--level, whose group has at least --min-stratum-size isolates (falling back
to shallower levels, then 'other'). Mixed or unknown calls form one group.

usage (from the old_tbprof root):
    python scripts/cmh.py --exclude-fold 0 \
        --splits /data/scratch/projects/punim1703/nn_tb/REVISION/results/essentials/splits.csv
    python scripts/cmh.py ... --min-fn-strata 2
"""

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.stats import chi2, fisher_exact
from tqdm import tqdm

OUT_COLS = ['snp', 'log_pval', 'log_or', 'fn_snp_prop', 'fn_no_snp_prop',
            'fn_count', 'n_fn_strata', 'crude_log_or']


# ---------------------------------------------------------------------------
# Sub-lineage strata
# ---------------------------------------------------------------------------

def _lineage_parts(call):
    """'lineage2.2.1' -> ['lineage2', '2', '1']; 'La1.8.1' -> ['La1', '8', '1'];
    mixed ('a;b') or unknown -> None."""
    call = str(call).strip()
    if not call or call.lower() in ('nan', 'unknown') or ';' in call:
        return None
    return call.split('.')


def assign_strata(lineage_by_run, max_level=3, min_size=50):
    """
    Map each run to a stratum label: its lineage truncated to the deepest
    level <= max_level whose group (over all isolates) has >= min_size members.
    """
    parts = {r: _lineage_parts(c) for r, c in lineage_by_run.items()}
    counts = {}
    for p in parts.values():
        if p is None:
            continue
        for k in range(1, max_level + 1):
            key = '.'.join(p[:k])
            counts[key] = counts.get(key, 0) + 1

    out = {}
    for r, p in parts.items():
        if p is None:
            out[r] = 'mixed_or_unknown'
            continue
        label = 'other'
        for k in range(min(max_level, len(p)), 0, -1):
            key = '.'.join(p[:k])
            if counts.get(key, 0) >= min_size:
                label = key
                break
        out[r] = label
    return out


# ---------------------------------------------------------------------------
# Loading rows
# ---------------------------------------------------------------------------

def load_rows(npz_path, splits, keep_folds):
    """Load an FN/TN matrix; keep training rows in keep_folds. Returns (matrix, runs, n_total)."""
    runs_path = npz_path.with_name(npz_path.name.replace('_snps.npz', '_runs.txt'))
    if not runs_path.exists():
        raise SystemExit(
            f"missing {runs_path}: run write_run_sidecars.py, or save the accession "
            f"of each row of {npz_path.name} (in row order) where it is built")

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
    return m[keep], runs[keep].to_numpy(), len(runs)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def _indicator(labels, levels):
    """Sparse (n_strata x n_rows) 0/1 matrix: which stratum each row is in."""
    idx = {s: i for i, s in enumerate(levels)}
    rows = np.array([idx[s] for s in labels], dtype=np.int64)
    return sp.csr_matrix((np.ones(len(labels)), (rows, np.arange(len(labels)))),
                         shape=(len(levels), len(labels)))


def cmh(fn, tn, fn_strata, tn_strata, cols):
    """
    Cochran-Mantel-Haenszel test (with continuity correction) and
    Mantel-Haenszel common odds ratio, for each column in `cols`, stratified
    by sub-lineage. Per stratum k:  a = FN carriers, b = FN non-carriers,
    c = TN carriers, d = TN non-carriers.
    Returns p, mh_or, n_fn_strata (arrays over cols).
    """
    levels = sorted(set(fn_strata) | set(tn_strata))
    S_fn = _indicator(fn_strata, levels)
    S_tn = _indicator(tn_strata, levels)

    a = np.asarray((S_fn @ fn[:, cols]).todense(), dtype=float)   # K x P
    c = np.asarray((S_tn @ tn[:, cols]).todense(), dtype=float)
    n1 = np.asarray(S_fn.sum(axis=1), dtype=float)                 # K x 1
    n0 = np.asarray(S_tn.sum(axis=1), dtype=float)
    N = n1 + n0
    b, d = n1 - a, n0 - c
    m1 = a + c

    ok = (n1 > 0) & (n0 > 0) & (N > 1)     # strata with both groups contribute
    with np.errstate(divide='ignore', invalid='ignore'):
        E = np.where(ok, n1 * m1 / N, 0.0)
        V = np.where(ok, n1 * n0 * m1 * (N - m1) / (N ** 2 * (N - 1)), 0.0)
        A, EA, VA = (np.where(ok, a, 0.0)).sum(0), E.sum(0), V.sum(0)
        stat = np.where(VA > 0, np.maximum(np.abs(A - EA) - 0.5, 0) ** 2 / VA, 0.0)
        p = np.where(VA > 0, chi2.sf(stat, 1), 1.0)

        num = np.where(ok, a * d / N, 0.0).sum(0)
        den = np.where(ok, b * c / N, 0.0).sum(0)
        mh_or = np.where(den > 0, num / den, np.where(num > 0, np.inf, np.nan))

    n_fn_strata = (a > 0).sum(0)
    return p, mh_or, n_fn_strata


def fisher(fn_ones, fn_zeros, tn_ones, tn_zeros):
    p = np.ones(len(fn_ones))
    odds = np.full(len(fn_ones), np.nan)
    for i in range(len(fn_ones)):
        res = fisher_exact([[fn_ones[i], fn_zeros[i]], [tn_ones[i], tn_zeros[i]]])
        p[i], odds[i] = res.pvalue, res.statistic
    return p, odds


def log10_or_inf(x):
    """log10 with +inf for inf, -inf for 0, nan for nan."""
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(np.isinf(x), np.inf, np.log10(x))


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--matrices', default='results/matrices')
    ap.add_argument('--snps', default='results/essentials/NEW_snps.txt')
    ap.add_argument('--splits', default='results/essentials/splits.csv')
    ap.add_argument('--lineage', default='resources/data/samples.csv',
                    help='CSV with run and lineage columns')
    ap.add_argument('--test', choices=['cmh', 'fisher'], default='cmh')
    ap.add_argument('--level', type=int, default=3,
                    help='deepest sub-lineage level for strata (lineage2.2.1 = 3)')
    ap.add_argument('--min-stratum-size', type=int, default=50,
                    help='smaller groups fall back to their parent lineage')
    ap.add_argument('--min-fn-strata', type=int, default=1,
                    help='keep only variants carried by FNs in at least this many '
                         'sub-lineages (2 removes single-cluster variants)')
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
          f"{args.exclude_fold or 'none'} excluded); test: {args.test}")

    if args.test == 'cmh':
        lin = pd.read_csv(args.lineage, usecols=['run', 'lineage'], dtype=str)
        strata = assign_strata(dict(zip(lin['run'], lin['lineage'])),
                               args.level, args.min_stratum_size)
        sizes = pd.Series(strata).value_counts()
        print(f"{len(sizes)} sub-lineage strata (level <= {args.level}, "
              f">= {args.min_stratum_size} isolates); largest: "
              f"{', '.join(f'{k} {v}' for k, v in sizes.head(5).items())}")

    drugs = sorted(p.name for p in matrices.iterdir() if p.is_dir())
    log = []

    for drug in tqdm(drugs, desc='drugs'):
        fn_path = matrices / drug / f'FN_{drug}_snps.npz'
        tn_path = matrices / drug / f'TN_{drug}_snps.npz'
        if not fn_path.exists() or not tn_path.exists():
            continue

        fn, fn_runs, n_fn_all = load_rows(fn_path, splits, keep_folds)
        tn, tn_runs, n_tn_all = load_rows(tn_path, splits, keep_folds)
        n_fn, n_tn = fn.shape[0], tn.shape[0]

        if fn.shape[1] != len(snps) or tn.shape[1] != len(snps):
            raise SystemExit(f"{drug}: matrix has {fn.shape[1]}/{tn.shape[1]} "
                             f"columns but {len(snps)} SNP names")

        fn_ones = np.asarray(fn.sum(axis=0)).ravel()
        tn_ones = np.asarray(tn.sum(axis=0)).ravel()
        # variants absent from both groups can't be enriched; skip them
        cols = np.flatnonzero(fn_ones + tn_ones > 0)
        fo, to = fn_ones[cols], tn_ones[cols]
        fz, tz = n_fn - fo, n_tn - to

        with np.errstate(divide='ignore', invalid='ignore'):
            crude_or = np.where(fz * to > 0, (fo * tz) / (fz * to),
                                np.where(fo * tz > 0, np.inf, np.nan))

        if args.test == 'cmh':
            missing = [r for r in np.concatenate([fn_runs, tn_runs]) if r not in strata]
            if missing:
                raise SystemExit(f"{drug}: {len(missing)} runs have no lineage in "
                                 f"{args.lineage} (e.g. {missing[:3]})")
            p, odds, n_strata = cmh(fn.tocsc(), tn.tocsc(),
                                    [strata[r] for r in fn_runs],
                                    [strata[r] for r in tn_runs], cols)
        else:
            p, odds = fisher(fo, fz, to, tz)
            n_strata = np.full(len(cols), -1)

        with np.errstate(invalid='ignore', divide='ignore'):
            fn_snp_prop = fo / (fo + to)
            fn_no_snp_prop = fz / (fz + tz)

        keep = p <= args.alpha
        if args.test == 'cmh' and args.min_fn_strata > 1:
            keep &= n_strata >= args.min_fn_strata

        with np.errstate(divide='ignore'):
            log_p = np.where(p > 0, np.log10(np.maximum(p, 1e-300)), -np.inf)
        df = pd.DataFrame({
            'snp': snps[cols][keep],
            'log_pval': np.round(log_p[keep], 3),
            'log_or': np.round(log10_or_inf(odds[keep]), 3),
            'fn_snp_prop': fn_snp_prop[keep],
            'fn_no_snp_prop': fn_no_snp_prop[keep],
            'fn_count': fo[keep].astype(int),
            'n_fn_strata': n_strata[keep].astype(int),
            'crude_log_or': np.round(log10_or_inf(crude_or[keep]), 3),
        }, columns=OUT_COLS)

        log.append({'drug': drug, 'fn_used': n_fn, 'fn_total': n_fn_all,
                    'tn_used': n_tn, 'tn_total': n_tn_all, 'n_written': len(df),
                    'n_p_le_1e-5_or_gt_10': int(((df['log_pval'] <= -5) &
                                                  (df['log_or'] > 1)).sum())})
        if df.empty:
            print(f"no enriched variants for {drug}")
        df.sort_values('log_pval', ascending=False).to_csv(
            matrices / drug / f'{drug}_enriched_snps.csv', index=False)

    pd.DataFrame(log).to_csv(matrices / 'enrichment_rows_used.csv', index=False)
    print(pd.DataFrame(log).to_string(index=False))


if __name__ == '__main__':
    main()
