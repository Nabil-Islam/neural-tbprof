"""
Fixed, accession-level data splits shared by every script.

The split is made once (scripts/make_splits.py) and saved as
results/essentials/splits.csv with one row per accession:

    run     split   fold    group
    ERR...  train   0..k-1  <group id>
    ERR...  test    -1      <group id>

Every training / tuning / evaluation script selects rows by accession ID
through `load_split`, so all variant sets (who / er / filt), the CV used
for hyperparameter tuning, and the TB-Profiler comparison use exactly the
same isolates, whatever the row order of a given matrix.

No tensorflow dependency.
"""

from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Iterative stratification (Sechidis, Tsoumakas & Vlahavas, 2011)
# ---------------------------------------------------------------------------

def _unit_label_counts(y, groups):
    """
    Collapse isolates into units (groups, or single isolates if groups is
    None) and count, per unit, resistant and susceptible isolates per drug.

    Returns
    -------
    unit_ids : array of unit keys, shape (n_units,)
    unit_of_row : int array mapping each isolate to its unit index
    counts : int array, shape (n_units, 2 * n_drugs)
        Columns [0, n_drugs) = resistant counts, [n_drugs, 2n) = susceptible.
        Stratifying on both keeps every drug's tested-isolate denominator
        balanced, not just its resistant isolates.
    sizes : int array, shape (n_units,)  isolates per unit
    """
    y = np.asarray(y, dtype=float)
    res = (y == 1).astype(int)
    sus = (y == 0).astype(int)
    per_row = np.hstack([res, sus])

    if groups is None:
        groups = np.arange(len(y))
    unit_ids, unit_of_row = np.unique(np.asarray(groups), return_inverse=True)

    counts = np.zeros((len(unit_ids), per_row.shape[1]), dtype=int)
    np.add.at(counts, unit_of_row, per_row)
    sizes = np.bincount(unit_of_row, minlength=len(unit_ids))
    return unit_ids, unit_of_row, counts, sizes


def iterative_stratify(y, proportions, groups=None, seed=42):
    """
    Assign isolates to len(proportions) subsets so that, for every drug,
    resistant and susceptible isolates are spread in the given proportions
    as closely as possible. Rare labels (e.g. bedaquiline resistance) are
    placed first, which is what plain random splitting gets wrong.

    If `groups` is given, all isolates of a group go to the same subset
    (e.g. transmission clusters or BioProjects), so near-identical genomes
    never straddle train and test.

    Parameters
    ----------
    y : array (n_isolates, n_drugs), 1 = R, 0 = S, NaN = untested
    proportions : list of floats summing to 1
    groups : array (n_isolates,) or None
    seed : int

    Returns
    -------
    assignment : int array (n_isolates,) with values in range(len(proportions))
    """
    rng = np.random.default_rng(seed)
    props = np.asarray(proportions, dtype=float)
    props = props / props.sum()
    k = len(props)

    unit_ids, unit_of_row, counts, sizes = _unit_label_counts(y, groups)
    n_units, n_labels = counts.shape

    desired_label = np.outer(props, counts.sum(0)).astype(float)  # (k, L)
    desired_size = props * sizes.sum()                            # (k,)
    unit_fold = np.full(n_units, -1)
    remaining = counts.copy()
    unassigned = np.ones(n_units, dtype=bool)

    # shuffle once so ties are broken randomly but reproducibly
    order = rng.permutation(n_units)

    while unassigned.any():
        label_totals = remaining[unassigned].sum(0)
        live = np.where(label_totals > 0)[0]

        if len(live) == 0:
            # units with no labels at all: fill by remaining size
            for u in order[unassigned[order]]:
                f = int(np.argmax(desired_size))
                unit_fold[u] = f
                desired_size[f] -= sizes[u]
                unassigned[u] = False
            break

        # rarest remaining label first
        lab = live[np.argmin(label_totals[live])]
        units = order[unassigned[order] & (counts[order, lab] > 0)]

        for u in units:
            want = desired_label[:, lab]
            cand = np.flatnonzero(want == want.max())
            if len(cand) > 1:  # tie-break on overall size still wanted
                cand = cand[desired_size[cand] == desired_size[cand].max()]
            f = int(rng.choice(cand)) if len(cand) > 1 else int(cand[0])

            unit_fold[u] = f
            desired_label[f] -= counts[u]
            desired_size[f] -= sizes[u]
            unassigned[u] = False

    return unit_fold[unit_of_row]


# ---------------------------------------------------------------------------
# Making and reading the split file
# ---------------------------------------------------------------------------

def make_splits(runs, y, test_frac=0.2, n_folds=5, groups=None, seed=42):
    """
    Hold out a stratified test set, then cut the remaining isolates into
    `n_folds` stratified CV folds. Returns a DataFrame (run, split, fold, group).
    """
    runs = np.asarray(runs)
    y = np.asarray(y, dtype=float)
    if groups is None:
        groups_arr = None
        group_col = runs
    else:
        groups_arr = np.asarray(groups)
        group_col = groups_arr

    is_test = iterative_stratify(
        y, [1 - test_frac, test_frac], groups=groups_arr, seed=seed) == 1

    fold = np.full(len(runs), -1)
    tr = ~is_test
    fold[tr] = iterative_stratify(
        y[tr], [1.0 / n_folds] * n_folds,
        groups=None if groups_arr is None else groups_arr[tr],
        seed=seed + 1)

    return pd.DataFrame({
        'run': runs,
        'split': np.where(is_test, 'test', 'train'),
        'fold': fold,
        'group': group_col,
    })


def load_split(splits_path, runs, val_fold=None):
    """
    Look up each matrix row's accession in splits.csv.

    Parameters
    ----------
    splits_path : path to splits.csv
    runs : array of accession IDs, in the row order of the matrix being split
    val_fold : int or None
        If given, CV fold `val_fold` of the training pool is returned as
        the validation set and the other folds as training.

    Returns
    -------
    dict of boolean row masks: 'test', 'train_pool' (all non-test), and,
    when val_fold is given, 'train' and 'val'. Also 'fold' (int array).
    """
    splits = pd.read_csv(splits_path, dtype={'run': str}).set_index('run')
    runs = pd.Index(np.asarray(runs).astype(str))

    missing = runs.difference(splits.index)
    if len(missing):
        raise ValueError(
            f"{len(missing)} accessions are not in {splits_path} "
            f"(e.g. {list(missing[:3])}). Re-run make_splits.py only if the "
            f"isolate set itself changed; otherwise fix the run IDs.")
    if runs.has_duplicates:
        raise ValueError("duplicate accession IDs in the matrix rows")

    s = splits.loc[runs]
    is_test = (s['split'] == 'test').to_numpy()
    fold = s['fold'].to_numpy().astype(int)

    out = {'test': is_test, 'train_pool': ~is_test, 'fold': fold}
    if val_fold is not None:
        n_folds = fold[~is_test].max() + 1
        if not 0 <= val_fold < n_folds:
            raise ValueError(f"val_fold must be in 0..{n_folds - 1}")
        out['val'] = (~is_test) & (fold == val_fold)
        out['train'] = (~is_test) & (fold != val_fold)
    return out


def split_summary(splits_df, y, drugs):
    """Resistant / tested counts per drug in each split and fold."""
    y = np.asarray(y, dtype=float)
    part = np.where(splits_df['split'] == 'test', 'test',
                    'fold' + splits_df['fold'].astype(str))
    rows = []
    for p in sorted(set(part), key=lambda s: (s != 'test', s)):
        m = part == p
        row = {'subset': p, 'n': int(m.sum())}
        for di, d in enumerate(drugs):
            yd = y[m, di]
            row[d] = f"{int(np.nansum(yd))}/{int((~np.isnan(yd)).sum())}"
        rows.append(row)
    return pd.DataFrame(rows)
