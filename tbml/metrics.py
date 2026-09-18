"""
Custom Keras metrics for masked multi-label classification.

(Plain-numpy confusion-matrix stats live in tbml.confusion, imported
here too for convenience/backward compatibility, since this module
used to hold both.)
"""

import tensorflow as tf
import tf_keras as keras

from tbml.confusion import confusion_stats  # noqa: F401  (re-exported)


class MaskedF1(keras.metrics.Metric):
    """F1 score that ignores NaN entries in y_true."""

    def __init__(self, name='masked_f1', threshold=0.5, **kwargs):
        super().__init__(name=name, **kwargs)
        self.threshold = threshold
        self.tp = self.add_weight(name='tp', initializer='zeros')
        self.fp = self.add_weight(name='fp', initializer='zeros')
        self.fn = self.add_weight(name='fn', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        mask = ~tf.math.is_nan(y_true)
        y_true_clean = tf.boolean_mask(y_true, mask)
        y_pred_clean = tf.boolean_mask(y_pred, mask)
        y_pred_bin = tf.cast(y_pred_clean > self.threshold, tf.float32)
        y_true_clean = tf.cast(y_true_clean, tf.float32)
        self.tp.assign_add(tf.reduce_sum(y_pred_bin * y_true_clean))
        self.fp.assign_add(tf.reduce_sum(y_pred_bin * (1 - y_true_clean)))
        self.fn.assign_add(tf.reduce_sum((1 - y_pred_bin) * y_true_clean))

    def result(self):
        prec = self.tp / (self.tp + self.fp + 1e-8)
        rec = self.tp / (self.tp + self.fn + 1e-8)
        return 2 * prec * rec / (prec + rec + 1e-8)

    def reset_state(self):
        self.tp.assign(0)
        self.fp.assign(0)
        self.fn.assign(0)


class MaskedAUC(keras.metrics.Metric):
    """AUC that ignores NaN entries in y_true."""

    def __init__(self, name='masked_auc', **kwargs):
        super().__init__(name=name, **kwargs)
        self.auc = keras.metrics.AUC()

    def update_state(self, y_true, y_pred, sample_weight=None):
        mask = tf.reshape(~tf.math.is_nan(y_true), [-1])
        y_true_flat = tf.boolean_mask(tf.reshape(y_true, [-1]), mask)
        y_pred_flat = tf.boolean_mask(tf.reshape(y_pred, [-1]), mask)
        self.auc.update_state(y_true_flat, y_pred_flat)

    def result(self):
        return self.auc.result()

    def reset_state(self):
        self.auc.reset_state()
