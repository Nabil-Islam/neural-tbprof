"""
Keras callbacks for tracking per-label (per-drug) metrics during training.
"""

import numpy as np
import tf_keras as keras


class PerDrugHistory(keras.callbacks.Callback):
    """
    Tracks loss/AUC/F1 per label (e.g. per drug) at the end of every epoch,
    using a fixed validation set. Useful when the built-in Keras metrics
    only report an aggregate across all labels.

    Parameters
    ----------
    x_val, y_val : validation inputs/labels (y_val may contain NaNs)
    labels : list of str
        Names for each output column, in column order.
    """

    def __init__(self, x_val, y_val, labels):
        super().__init__()
        self.x_val = x_val
        self.y_val = y_val
        self.labels = labels
        self.history = {label: {'loss': [], 'auc': [], 'f1': []}
                         for label in labels}
        self._auc_metric = keras.metrics.AUC()

    def on_epoch_end(self, epoch, logs=None):
        y_pred = self.model.predict(self.x_val, verbose=0)
        for i, label in enumerate(self.labels):
            mask = ~np.isnan(self.y_val[:, i])
            yt = self.y_val[mask, i]
            yp = y_pred[mask, i]

            if len(yt) == 0 or len(np.unique(yt)) < 2:
                self.history[label]['loss'].append(np.nan)
                self.history[label]['auc'].append(np.nan)
                self.history[label]['f1'].append(np.nan)
                continue

            loss = -np.mean(yt * np.log(yp + 1e-8) +
                             (1 - yt) * np.log(1 - yp + 1e-8))

            self._auc_metric.reset_state()
            self._auc_metric.update_state(yt, yp)
            auc_val = self._auc_metric.result().numpy()

            yb = (yp > 0.5).astype(float)
            tp = ((yb == 1) & (yt == 1)).sum()
            fp = ((yb == 1) & (yt == 0)).sum()
            fn = ((yb == 0) & (yt == 1)).sum()
            prec = tp / (tp + fp + 1e-8)
            rec = tp / (tp + fn + 1e-8)
            f1 = 2 * prec * rec / (prec + rec + 1e-8)

            self.history[label]['loss'].append(loss)
            self.history[label]['auc'].append(auc_val)
            self.history[label]['f1'].append(f1)
