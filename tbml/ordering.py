"""
Helpers for reconciling column orderings between the label matrix and a
model's output.

Motivation
----------
The cross-resistance model emits its outputs as
Concatenate([non_cr_drugs, cr_drugs]), which is NOT the alphabetical
order the label matrix (labelled_vcfs.csv) uses. Feeding those straight
into per-drug metrics compares the wrong drug against the wrong labels,
silently, with no error raised. These helpers make the mapping explicit.

No tensorflow dependency, so this is cheap to import and unit-test.
"""

import numpy as np


def cross_resistance_output_order(non_cr_drugs, cr_pairs):
    """
    The drug order a cross-resistance model actually outputs, given how
    the graph concatenates [r_out, cr_out].

    Parameters
    ----------
    non_cr_drugs : sequence of str
        Drugs handled by the plain r_layer, in the column order of the
        {name}_WEIGHTS_AUG.npz matrix used to initialise that layer.
    cr_pairs : dict
        Mapping drug -> its cross-resistance partner (e.g. cr_dic).
        Key order defines the cr_layer / cr_pred column order.

    Returns
    -------
    list of str, length len(non_cr_drugs) + len(cr_pairs)
    """
    return list(non_cr_drugs) + list(cr_pairs.keys())


def partner_permutation(cr_pairs):
    """
    Derive the gather index list that swaps each cross-resistance drug's
    column with its partner's, from the pair mapping itself.

    For cr_pairs with keys [lfx, mfx, bdq, cfz, inh, eto] this returns
    [1, 0, 3, 2, 5, 4] -- i.e. the hardcoded p_idx, but derived, so it
    stays correct if the pair dict is ever reordered or extended.
    """
    cr_drugs = list(cr_pairs.keys())
    missing = [p for p in cr_pairs.values() if p not in cr_drugs]
    if missing:
        raise ValueError(
            f"cross-resistance partners not present as keys: {missing}. "
            "cr_pairs must be symmetric (both directions listed)."
        )
    return [cr_drugs.index(cr_pairs[d]) for d in cr_drugs]


def permutation_to(from_order, to_order):
    """
    Index array `perm` such that `arr[:, perm]` converts a matrix whose
    columns are in `from_order` into one whose columns are in `to_order`.

    Raises if the two orders aren't the same set of names, which catches
    typos and missing drugs up front rather than at metric time.
    """
    from_order = list(from_order)
    to_order = list(to_order)

    if set(from_order) != set(to_order):
        only_from = sorted(set(from_order) - set(to_order))
        only_to = sorted(set(to_order) - set(from_order))
        raise ValueError(
            "column orders do not describe the same set of drugs.\n"
            f"  only in from_order: {only_from}\n"
            f"  only in to_order:   {only_to}"
        )
    if len(from_order) != len(set(from_order)):
        raise ValueError("from_order contains duplicate drug names")

    index = {name: i for i, name in enumerate(from_order)}
    return np.array([index[name] for name in to_order], dtype=int)


def reorder_columns(arr, from_order, to_order):
    """
    Reorder the columns of a 2-D array from one drug ordering to another.

    Example
    -------
    >>> y_model_order = reorder_columns(y, label_drugs, model_output_drugs)
    """
    perm = permutation_to(from_order, to_order)
    return np.asarray(arr)[:, perm]


def assert_output_order(model, expected_order):
    """
    Sanity check that a model's output width matches the expected drug
    list length. Width is all we can verify automatically -- the actual
    *names* depend on how the weight matrices were built, so this is a
    guard rail, not a proof.
    """
    n_out = model.output_shape[-1]
    if n_out != len(expected_order):
        raise ValueError(
            f"model outputs {n_out} columns but expected_order lists "
            f"{len(expected_order)} drugs: {list(expected_order)}"
        )
    return True
