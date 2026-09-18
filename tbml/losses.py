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


def make_masked_weighted_bce(pos_weight_tf):
    """
    Factory for a masked, per-label-weighted binary crossentropy loss.

    Parameters
    ----------
    pos_weight_tf : tf.Tensor
        Per-label positive-class weight, shape (n_labels,). Typically
        derived from class imbalance, e.g.:

            pos_counts = np.nansum(y_train, axis=0)
            neg_counts = np.nansum(1 - y_train, axis=0)
            pos_weight = np.sqrt(neg_counts / (pos_counts + 1e-8))
            pos_weight_tf = tf.constant(pos_weight, dtype=tf.float32)

    Returns
    -------
    A loss function usable directly in model.compile(loss=...).
    """

    def masked_weighted_bce(y_true, y_pred):
        mask = tf.cast(~tf.math.is_nan(y_true), tf.float32)
        y_true_clean = tf.where(tf.math.is_nan(y_true), tf.zeros_like(y_true), y_true)
        weights = y_true_clean * pos_weight_tf + (1 - y_true_clean)
        loss = keras.backend.binary_crossentropy(y_true_clean, y_pred) * weights * mask
        return tf.reduce_sum(loss) / (tf.reduce_sum(mask) + 1e-8)

    return masked_weighted_bce


def compute_pos_weight(y_train):
    """
    Compute sqrt(neg/pos) per-label positive-class weights from a
    (possibly NaN-containing) label matrix, ready to feed into
    make_masked_weighted_bce.
    """
    import numpy as np

    pos_counts = np.nansum(y_train, axis=0)
    neg_counts = np.nansum(1 - y_train, axis=0)
    pos_weight = np.sqrt(neg_counts / (pos_counts + 1e-8))
    return tf.constant(pos_weight, dtype=tf.float32)
