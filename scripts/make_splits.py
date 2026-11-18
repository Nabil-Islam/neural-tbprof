"""
Make the one fixed train/test split (+ CV folds) used by every script.

Run once, from the repository root, after label_data.py:

    python scripts/make_splits.py
    python scripts/make_splits.py --groups results/essentials/clusters.csv

Writes results/essentials/splits.csv (run, split, fold, group) and a
per-drug count table so you can check rare resistance is spread evenly.

The split is at the accession level, so it is independent of the variant
set: who / er / filt matrices, the loss-weight tuning, and the TB-Profiler
comparison all use the same isolates. Commit splits.csv; never regenerate
it after looking at results (it refuses to overwrite without --force).

--groups : optional CSV with columns run,group (e.g. SNP-distance
           transmission clusters, lineage, or BioProject). Every isolate
           in a group goes to the same side of the split and the same fold.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from tbml.splits import make_splits, split_summary


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--labels', default='results/essentials/labelled_vcfs.csv')
    ap.add_argument('--out', default='results/essentials/splits.csv')
    ap.add_argument('--groups', default=None,
                    help='optional CSV with columns run,group')
    ap.add_argument('--test-frac', type=float, default=0.2)
    ap.add_argument('--folds', type=int, default=5)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--force', action='store_true',
                    help='overwrite an existing splits file')
    args = ap.parse_args()

    out = Path(args.out)
    if out.exists() and not args.force:
        raise SystemExit(
            f"{out} already exists. The split must stay fixed once results "
            f"have been looked at; pass --force only if the isolate set changed.")

    labels = pd.read_csv(args.labels)
    runs = labels['run'].astype(str).to_numpy()
    drugs = [c for c in labels.columns if c != 'run']
    y = labels[drugs].to_numpy(dtype=float)

    if pd.Index(runs).has_duplicates:
        raise SystemExit("duplicate run IDs in the label file")

    groups = None
    if args.groups:
        g = pd.read_csv(args.groups, dtype={'run': str}).set_index('run')['group']
        missing = pd.Index(runs).difference(g.index)
        if len(missing):
            # isolates with no cluster are their own singleton group
            print(f"{len(missing)} isolates have no group; treating each as its own")
        groups = np.array([str(g.get(r, f"__single__{r}")) for r in runs])
        sizes = pd.Series(groups).value_counts()
        print(f"{len(sizes)} groups; largest = {sizes.iloc[0]} isolates "
              f"({sizes.iloc[0] / len(runs):.1%} of data)")

    splits = make_splits(runs, y, test_frac=args.test_frac, n_folds=args.folds,
                         groups=groups, seed=args.seed)

    out.parent.mkdir(parents=True, exist_ok=True)
    splits.to_csv(out, index=False)

    summary = split_summary(splits, y, drugs)
    summary.to_csv(out.with_name(out.stem + '_summary.csv'), index=False)

    n_test = (splits['split'] == 'test').sum()
    print(f"Saved {out}: {len(splits)} isolates, {n_test} test "
          f"({n_test / len(splits):.1%}), {args.folds} CV folds in train")
    print("Resistant / tested per drug:")
    with pd.option_context('display.width', 250, 'display.max_columns', None):
        print(summary.to_string(index=False))


if __name__ == '__main__':
    main()
