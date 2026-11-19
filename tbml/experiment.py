"""
Shared setup for the anchored single-layer models, used by both
scripts/tune_loss.py and scripts/train_new.py so that tuning and the final
training run build, initialise and train the model identically.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

DRUGS = ['amikacin', 'bedaquiline', 'capreomycin', 'clofazimine', 'delamanid',
         'ethambutol', 'ethionamide', 'isoniazid', 'kanamycin', 'levofloxacin',
         'linezolid', 'moxifloxacin', 'pyrazinamide', 'rifampicin', 'streptomycin']

VARIANT_SETS = {
    'who': ('who', 'WHO'),
    'er': ('expert_rules', 'Expert rules'),
    'filt': ('filtered_expert_rules', 'Filtered expert rules'),
}

KERNEL_SCALE = 10.0     # catalogue weights are scaled up before training
ANCHOR_STRENGTH = 1e-5
LEARNING_RATE = 1e-2      # converges in ~15 epochs to the same val loss as 1e-3 in ~300 (fold-0 diagnostic)
BATCH_SIZE = 64
MAX_EPOCHS = 100

# Early stopping / LR schedule watch each setting's own (weighted) loss on
# the validation data. Not the unweighted BCE: a reweighted model is
# deliberately miscalibrated, so its unweighted BCE rises while it is still
# improving, and stopping on it cuts every non-default setting off after
# ~1 epoch. The unweighted BCE is still logged as a metric.
MONITOR = 'val_loss'


def load_variant_set(results, arg, load_kernel=True):
    """
    Load features, labels, row IDs and the catalogue kernel for one variant set.
    Rows of X and y both follow vcf_list.txt order (make_index.py, label_data.py).
    """
    if arg not in VARIANT_SETS:
        raise ValueError(f"unknown variant set '{arg}', expected one of {list(VARIANT_SETS)}")
    import h5py

    results = Path(results)
    name, tname = VARIANT_SETS[arg]

    X = sp.load_npz(results / f'matrices/{name}/{name}_AUG_matrix.npz').toarray().astype(np.float32)

    labels = pd.read_csv(results / 'essentials/labelled_vcfs.csv')
    runs = labels['run'].astype(str).to_numpy()
    labels = labels.drop(columns=['run'])
    if list(labels.columns) != DRUGS:
        raise ValueError(f"label columns {list(labels.columns)} != DRUGS")
    y = labels.to_numpy().astype(np.float32)

    if X.shape[0] != len(runs):
        raise ValueError(f"matrix has {X.shape[0]} rows but {len(runs)} labelled runs")

    kernel = None
    if load_kernel:   # the cross-resistance model loads its own starting weights
        with h5py.File(str(results / f'models/naiive_models/{name}_naiive_model.h5'), 'r') as f:
            kernel = f['model_weights/large_layer/large_layer/kernel'][:] * KERNEL_SCALE

    return dict(X=X, y=y, runs=runs, kernel=kernel, name=name, tname=tname)


def initial_bias(y_tr, pos_weight, neg_weight=1.0):
    """
    Logit of each drug's resistance prevalence, shifted by the class-weight
    ratio (the optimum of a weighted BCE with no variants present).
    """
    p = np.nanmean(y_tr, axis=0)
    return np.log(p / (1 - p)) + np.log(np.asarray(pos_weight) / neg_weight)


def build_and_compile(kernel, y_tr, alpha=0.5, neg_weight=1.0, fp_gamma=0.0,
                      extra_metrics=(), learning_rate=LEARNING_RATE):
    """Anchored dense model, catalogue-initialised, compiled with the weighted loss."""
    import tf_keras as keras
    from tbml.losses import (compute_pos_weight, make_masked_weighted_bce,
                             masked_binary_crossentropy)
    from tbml.models import build_anchored_dense_model

    pw = compute_pos_weight(y_tr, alpha=alpha)
    model = build_anchored_dense_model(
        input_dim=kernel.shape[0], n_outputs=kernel.shape[1],
        kernel_init=kernel, bias_init=initial_bias(y_tr, pw.numpy(), neg_weight),
        layer_name='large_layer', activation='sigmoid',
        anchor_strength=ANCHOR_STRENGTH)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=make_masked_weighted_bce(pw, neg_weight=neg_weight, fp_gamma=fp_gamma,
                                      normalize='weight'),
        metrics=[masked_binary_crossentropy, *extra_metrics])
    return model


# An epoch only counts as an improvement if the validation loss drops by at
# least this much. Without it, a loss that has plateaued (0.2929 -> 0.29289)
# keeps "improving" in the 5th decimal and early stopping never triggers.
MIN_DELTA = 1e-4


def training_callbacks(early_stopping_patience=8, lr_patience=3, min_delta=MIN_DELTA):
    import tf_keras as keras
    return [
        keras.callbacks.EarlyStopping(monitor=MONITOR, mode='min',
                                      patience=early_stopping_patience,
                                      min_delta=min_delta,
                                      restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor=MONITOR, mode='min', factor=0.2,
                                          patience=lr_patience, min_lr=1e-6),
    ]


def per_drug_scores(y_true, y_score, drugs, target_spec):
    """
    For each drug: threshold at the target specificity on these predictions,
    and the sensitivity / specificity there, plus AUC and partial AUC
    (FPR <= 0.05, McClish-standardised as in sklearn).
    """
    from sklearn.metrics import roc_auc_score
    from tbml.confusion import confusion_stats
    from tbml.thresholds import threshold_at_spec

    rows = []
    for di, d in enumerate(drugs):
        m = ~np.isnan(y_true[:, di])
        yt, ys = y_true[m, di], y_score[m, di]
        row = {'drug': d, 'n': int(m.sum()), 'n_resistant': int(yt.sum()),
               'target_spec': target_spec[d]}
        if len(np.unique(yt)) < 2:
            rows.append(row)
            continue
        thr = threshold_at_spec(yt, ys, target_spec[d])
        st = confusion_stats(yt, ys, thr)
        row.update(threshold=thr, sens_at_spec=float(st['sensitivity']),
                   spec=float(st['specificity']),
                   auc=float(roc_auc_score(yt, ys)),
                   pauc=float(roc_auc_score(yt, ys, max_fpr=0.05)))
        rows.append(row)
    return pd.DataFrame(rows)
