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
    exact_p       True if log_pval is exact rather than the CMH approximation

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

3. Co-resistance (--co-resistance). False negatives for many drugs are
   mostly MDR isolates while true negatives mostly aren't (e.g. ethambutol:
   60% vs 6%), so variants that travel with MDR (compensatory rpoC/rpoA
   mutations, hitchhikers) look enriched within every lineage. With
   --co-resistance each lineage stratum is split by rifampicin and
   isoniazid phenotype (R / S / untested), so a variant must be enriched
   in FNs compared with isolates of the same lineage and the same RIF/INH
   status. For rifampicin itself only isoniazid is used, and vice versa,
   so the outcome being tested is never a stratifier. Phenotypes come from
   --phenotypes (training isolates only, as the rows already are).
   n_fn_strata still counts lineages, not lineage x phenotype groups.

Sub-lineages come from the `lineage` column of resources/data/samples.csv
(e.g. lineage2.2.1). Each isolate is placed at the deepest level, up to
--level, whose group has at least --min-stratum-size isolates (falling back
to shallower levels, then 'other'). Mixed or unknown calls form one group.

usage (from the old_tbprof root):
    python scripts/cmh.py --exclude-fold 0 \
        --splits /data/scratch/projects/punim1703/nn_tb/REVISION/results/essentials/splits.csv
    python scripts/cmh.py ... --min-fn-strata 2
    python scripts/cmh.py ... --min-fn-strata 2 --co-resistance   # also control for MDR
"""

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.stats import chi2, fisher_exact, hypergeom
from tqdm import tqdm

OUT_COLS = ['snp', 'log_pval', 'log_or', 'fn_snp_prop', 'fn_no_snp_prop',
            'fn_count', 'n_fn_strata', 'crude_log_or', 'exact_p']


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


def exact_stratified_p(a, n1, n0, m1):
    """
    Exact two-sided p-value for the CMH null (no association in any stratum),
    for one variant. Given each stratum's margins, the number of carriers
    that are FN is hypergeometric; the total over strata is the convolution
    of those. p = sum of probabilities of totals no more likely than the
    observed one (as Fisher's exact test does for a single table).
    a, n1, n0, m1: per-stratum arrays (FN carriers, FN, TN, carriers).
    """
    pmf = np.array([1.0])
    for ak, n1k, n0k, m1k in zip(a, n1, n0, m1):
        if m1k == 0:
            continue
        lo, hi = int(max(0, m1k - n0k)), int(min(m1k, n1k))
        x = np.arange(lo, hi + 1)
        pk = hypergeom.pmf(x, int(n1k + n0k), int(n1k), int(m1k))
        pk = np.concatenate([np.zeros(lo), pk])        # index = number of FN carriers
        pmf = np.convolve(pmf, pk)
    obs = int(round(a.sum()))
    if obs >= len(pmf):
        return 0.0
    return float(min(1.0, pmf[pmf <= pmf[obs] * (1 + 1e-7)].sum()))


def cmh(fn, tn, fn_strata, tn_strata, cols, fn_count_labels=None,
        alpha=0.05, exact_max_carriers=200):
    """
    Cochran-Mantel-Haenszel test (with continuity correction) and
    Mantel-Haenszel common odds ratio, for each column in `cols`, stratified
    by sub-lineage. Per stratum k:  a = FN carriers, b = FN non-carriers,
    c = TN carriers, d = TN non-carriers.
    fn_count_labels: labels used for n_fn_strata (default: fn_strata), so
    the count can be over lineages when the test strata are finer.
    The CMH chi-square approximation is badly over-confident for rare
    variants (2 carriers, both FN, can get p ~ 1e-13 instead of ~ 1e-3),
    so variants with p <= alpha and at most exact_max_carriers carriers get
    an exact p-value instead (exact_max_carriers=0 disables this).
    Returns p, mh_or, n_fn_strata, is_exact (arrays over cols).
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

    is_exact = np.zeros(len(cols), dtype=bool)
    if exact_max_carriers > 0:
        sel = ok.ravel()                       # strata with both FN and TN
        carriers = m1[sel].sum(0)
        todo = np.flatnonzero((p <= alpha) & (carriers <= exact_max_carriers))
        n1s, n0s = n1.ravel()[sel], n0.ravel()[sel]
        for j in todo:
            p[j] = exact_stratified_p(a[sel, j], n1s, n0s, m1[sel, j])
            is_exact[j] = True

    if fn_count_labels is None:
        n_fn_strata = (a > 0).sum(0)
    else:
        lv = sorted(set(fn_count_labels))
        a_l = (_indicator(fn_count_labels, lv) @ fn[:, cols]).tocsr()
        n_fn_strata = np.asarray((a_l > 0).sum(0)).ravel()
    return p, mh_or, n_fn_strata, is_exact


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
    ap.add_argument('--exact-max-carriers', type=int, default=200,
                    help='exact p-value for variants with at most this many carriers '
                         '(the CMH approximation is over-confident for rare variants); 0 = off')
    ap.add_argument('--co-resistance', action='store_true',
                    help='also stratify by rifampicin / isoniazid phenotype (controls for MDR)')
    ap.add_argument('--co-drugs', nargs='+', default=['rifampicin', 'isoniazid'],
                    help='phenotypes used by --co-resistance')
    ap.add_argument('--phenotypes', default='resources/data/samples.csv',
                    help='wide CSV: run + one R/S column per drug (for --co-resistance)')
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
        if args.co_resistance:
            ph = pd.read_csv(args.phenotypes, usecols=['run', *args.co_drugs],
                             dtype=str).set_index('run')
            print(f"co-resistance strata: lineage x {' x '.join(args.co_drugs)} phenotype")
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
            fn_lin = [strata[r] for r in fn_runs]
            tn_lin = [strata[r] for r in tn_runs]
            if args.co_resistance:
                co = [c for c in args.co_drugs if c != drug]   # never the drug being tested

                def label(run, lin_label):
                    vals = []
                    for c in co:
                        v = ph[c].get(run) if run in ph.index else None
                        vals.append(f"{c[:3]}{v if v in ('R', 'S') else 'NA'}")
                    return lin_label + '|' + ','.join(vals)
                fn_test = [label(r, l) for r, l in zip(fn_runs, fn_lin)]
                tn_test = [label(r, l) for r, l in zip(tn_runs, tn_lin)]
            else:
                fn_test, tn_test = fn_lin, tn_lin
            p, odds, n_strata, is_exact = cmh(
                fn.tocsc(), tn.tocsc(), fn_test, tn_test, cols, fn_count_labels=fn_lin,
                alpha=args.alpha, exact_max_carriers=args.exact_max_carriers)
        else:
            p, odds = fisher(fo, fz, to, tz)
            n_strata = np.full(len(cols), -1)
            is_exact = np.ones(len(cols), dtype=bool)

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
            'exact_p': is_exact[keep],
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
