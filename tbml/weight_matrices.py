"""
Building the sparse warm-start weight matrices for the naiive models.

This is the reusable core of scripts/new_layer/cr_w.py. Pulling it out
here means the *same* drug-ordering logic is used to build the weights
and to interpret the model's outputs (tbml.ordering), rather than the
two being kept in sync by hand.
"""

import numpy as np
import scipy.sparse as sp


def build_rule_weight_matrix(drug_order, snp_lookup, rules):
    """
    Sparse (n_drugs, n_snps) indicator matrix: 1 where a SNP is a
    resistance rule for a drug.

    Parameters
    ----------
    drug_order : sequence of str
        Row order. SNPs are looked up per drug from `rules`.
    snp_lookup : dict
        SNP name -> column index.
    rules : dict
        drug -> iterable of SNP names.
    """
    rows, cols = [], []
    for row_idx, drug in enumerate(drug_order):
        for snp in rules.get(drug, []):
            col_idx = snp_lookup.get(snp)
            if col_idx is not None:
                rows.append(row_idx)
                cols.append(col_idx)

    data = np.ones(len(rows), dtype=np.int8)
    return sp.coo_matrix(
        (data, (np.array(rows, dtype=np.int32), np.array(cols, dtype=np.int32))),
        shape=(len(drug_order), len(snp_lookup)),
    )


def build_cr_plus_signal_matrix(cr_drug_order, snp_lookup, rules,
                                 signal_weight=1):
    """
    Sparse (n_cr, n_snps + n_cr) matrix for the cr_pred layer, whose
    input is Concatenate([snp_vec, cr_reorder]).

    The trailing n_cr columns are the cross-resistance signals. Because
    GatherColumns has already swapped each drug with its partner, slot
    `i` of that block carries the partner signal for the drug in row
    `i` -- so the weight goes at column `n_snps + i`, i.e. on the
    diagonal of the trailing block.

    NOTE: a fixed column such as `n_snps + n_cr` is both out of bounds
    (valid indices stop at n_snps + n_cr - 1) and wrong in principle,
    since it would point every CR drug at the same single slot.

    Parameters
    ----------
    cr_drug_order : sequence of str
        Row order; also defines the slot order of the signal block.
    snp_lookup : dict
        SNP name -> column index.
    rules : dict
        drug -> iterable of SNP names.
    signal_weight : int or float
        Initial weight placed on the partner signal.
    """
    n_snps = len(snp_lookup)
    n_cr = len(cr_drug_order)

    rows, cols, vals = [], [], []
    for row_idx, drug in enumerate(cr_drug_order):
        for snp in rules.get(drug, []):
            col_idx = snp_lookup.get(snp)
            if col_idx is not None:
                rows.append(row_idx)
                cols.append(col_idx)
                vals.append(1)
        # partner signal for THIS drug, in the swapped block
        rows.append(row_idx)
        cols.append(n_snps + row_idx)
        vals.append(signal_weight)

    return sp.coo_matrix(
        (np.array(vals, dtype=np.int8),
         (np.array(rows, dtype=np.int32), np.array(cols, dtype=np.int32))),
        shape=(n_cr, n_snps + n_cr),
    )


def sparse_to_kernel(matrix):
    """
    Convert a saved (n_units, n_inputs) sparse weight matrix into the
    dense (n_inputs, n_units) kernel Keras expects, via reorder+transpose.
    """
    m = matrix.tocoo()
    dense = np.zeros(m.shape, dtype=np.float32)
    dense[m.row, m.col] = m.data
    return dense.T
