"""
Custom loss functions for multi-label training where some labels are
missing (encoded as NaN) for a given sample.
"""

import tensorflow as tf
import tf_keras as keras


def masked_binary_crossentropy(y_true, y_pred):
    """
    Binary crossentropy that ignores NaN entries in y_true.

    Use this when some (sample, label) pairs are missing and shouldn't
    contribute to the loss at all.
    """
    mask = tf.cast(~tf.math.is_nan(y_true), tf.float32)
    y_true_clean = tf.where(tf.math.is_nan(y_true), tf.zeros_like(y_true), y_true)
    loss = keras.backend.binary_crossentropy(y_true_clean, y_pred)
    loss = loss * mask
    return tf.reduce_sum(loss) / (tf.reduce_sum(mask) + 1e-8)


def make_masked_weighted_bce(pos_weight_tf, neg_weight=1.0, fp_gamma=0.0,
                             normalize='count'):
    """
    Factory for a masked, per-label-weighted binary crossentropy loss.

    Parameters
    ----------
    pos_weight_tf : tf.Tensor
        Per-label weight on resistant (positive) isolates, shape (n_labels,),
        e.g. from compute_pos_weight(). >1 makes missed resistance (FN) costlier.
    neg_weight : float
        Weight on susceptible (negative) isolates, shared by all labels.
        >1 makes false positives costlier overall.
    fp_gamma : float
        >0 redistributes the susceptible isolates' weight towards those the
        model scores as resistant (the actual false positives), in proportion
        to p**fp_gamma. The weights are rescaled so their mean per label stays
        `neg_weight` -- unlike focal loss, this focuses the FP penalty without
        shrinking the total weight on susceptible isolates.
    normalize : 'count' or 'weight'
        'count' divides by the number of observed labels (original behaviour).
        'weight' divides by the total weight, so changing the weights changes
        the balance between isolates but not the loss scale relative to the
        anchor regulariser. Use 'weight' when comparing weight settings.

    With the defaults this is exactly the original loss.

    Returns
    -------
    A loss function usable directly in model.compile(loss=...).
    """
    if normalize not in ('count', 'weight'):
        raise ValueError("normalize must be 'count' or 'weight'")
    neg_weight = float(neg_weight)
    fp_gamma = float(fp_gamma)

    def masked_weighted_bce(y_true, y_pred):
        mask = tf.cast(~tf.math.is_nan(y_true), tf.float32)
        y = tf.where(tf.math.is_nan(y_true), tf.zeros_like(y_true), y_true)
        neg = (1 - y) * mask

        w_neg = neg_weight * neg
        if fp_gamma > 0:
            p = tf.stop_gradient(tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7))
            focus = tf.pow(p, fp_gamma) * neg
            # rescale so the mean weight over susceptible isolates (per label,
            # in this batch) stays neg_weight
            mean_focus = tf.math.divide_no_nan(
                tf.reduce_sum(focus, axis=0, keepdims=True),
                tf.reduce_sum(neg, axis=0, keepdims=True))
            w_neg = neg_weight * tf.math.divide_no_nan(focus, mean_focus)

        weights = (y * pos_weight_tf) * mask + w_neg
        loss = keras.backend.binary_crossentropy(y, y_pred) * weights
        if normalize == 'weight':
            denom = tf.reduce_sum(tf.stop_gradient(weights))
        else:
            denom = tf.reduce_sum(mask)
        return tf.reduce_sum(loss) / (denom + 1e-8)

    return masked_weighted_bce


def compute_pos_weight(y_train, alpha=0.5):
    """
    Per-label positive-class weights (neg/pos)**alpha from a (possibly
    NaN-containing) label matrix, ready for make_masked_weighted_bce.

    alpha = 0.5 : sqrt(neg/pos), the original behaviour
    alpha = 0   : no class weighting (all 1)
    alpha < 0   : resistant isolates weighted *less* than susceptible
    """
    import numpy as np

    pos_counts = np.nansum(y_train, axis=0)
    neg_counts = np.nansum(1 - y_train, axis=0)
    pos_weight = (neg_counts / (pos_counts + 1e-8)) ** alpha
    return tf.constant(pos_weight, dtype=tf.float32)
