"""
Train the cross-resistance model.

The key difference from scripts/train.py: this architecture emits its
outputs as [non-CR drugs, CR drugs], which is NOT the alphabetical order
of the label matrix. We reorder the labels to match the model, train,
then reorder predictions back to alphabetical for reporting -- so every
CSV and figure downstream stays in the familiar drug order.
"""

import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import tensorflow as tf
import tf_keras as keras

from tbml.plotting import set_plot_style, plot_learning_curves, plot_confusion_grid
from tbml.losses import make_masked_weighted_bce, compute_pos_weight
from tbml.metrics import MaskedF1, MaskedAUC
from tbml.confusion import confusion_stats
from tbml.callbacks import PerDrugHistory
from tbml.thresholds import pick_threshold
from tbml.predictions import save_prediction_tables
from tbml.models import build_cross_resistance_model, strip_anchor_regularizers
from tbml.weights import load_layer_weights
from tbml.ordering import (cross_resistance_output_order, reorder_columns,
                           assert_output_order)

set_plot_style()

arg = sys.argv[1]
NAME_MAP = {
    'who': ('who', 'WHO'),
    'er': ('expert_rules', 'Expert rules'),
    'filt': ('filtered_expert_rules', 'Filtered expert rules'),
}
name, tname = NAME_MAP[arg]

SCRIPTS = Path.cwd()
RESULTS = SCRIPTS / 'results'
FIGURES = RESULTS / 'figures'
FIGURES.mkdir(parents=True, exist_ok=True)
METRICS = RESULTS / 'metrics'
METRICS.mkdir(parents=True, exist_ok=True)
OUT = METRICS / 'cr'        # per-run outputs; naiive_perf_metrics.csv stays in METRICS
OUT.mkdir(parents=True, exist_ok=True)

# --- Drug definitions -------------------------------------------------
# Alphabetical order == the column order of labelled_vcfs.csv
label_drugs = ['amikacin', 'bedaquiline', 'capreomycin', 'clofazimine', 'delamanid',
               'ethambutol', 'ethionamide', 'isoniazid', 'kanamycin', 'levofloxacin',
               'linezolid', 'moxifloxacin', 'pyrazinamide', 'rifampicin', 'streptomycin']

cr_pairs = {
    "levofloxacin": "moxifloxacin",
    "moxifloxacin": "levofloxacin",
    "bedaquiline":  "clofazimine",
    "clofazimine":  "bedaquiline",
    "isoniazid":    "ethionamide",
    "ethionamide":  "isoniazid",
}

# Confirmed against scripts/new_layer/cr_w.py: that script builds the
# r_layer matrix by iterating `drugs` (the snps JSON keys with the six CR
# drugs removed, order preserved), which comes out alphabetical. So the
# non-CR block is simply the alphabetical list minus the CR drugs.
non_cr_drugs = [d for d in label_drugs if d not in cr_pairs]

# The order the model actually outputs
model_drugs = cross_resistance_output_order(non_cr_drugs, cr_pairs)
print("model output order:", model_drugs)

# --- Load features ----------------------------------------------------
matrix = sp.load_npz(RESULTS / f'matrices/{name}/{name}_AUG_matrix.npz')
X = matrix.toarray().astype(np.float32)   # dense for model.fit()

# --- Load labels, then reorder to MODEL order -------------------------
label_matrix = pd.read_csv(RESULTS / 'essentials/labelled_vcfs.csv')
# keep sample ids before dropping
runs = label_matrix['run'].to_numpy()
label_matrix = label_matrix.drop(columns=['run'])
y_label_order = label_matrix.to_numpy().astype(np.float32)
y = reorder_columns(y_label_order, label_drugs, model_drugs)

# --- 80/20 split ------------------------------------------------------
rng = np.random.default_rng(42)
idx = rng.permutation(len(X))
split = int(0.8 * len(X))
x_train, x_test = X[idx[:split]], X[idx[split:]]
y_train, y_test = y[idx[:split]], y[idx[split:]]
runs_test = runs[idx[split:]]                  # same shuffle as x_test/y_test

# --- Warm-start from the naiive CR model ------------------------------
anchor = load_layer_weights(
    RESULTS / f'models/naiive_models/{name}_CR_naiive_model.h5',
    ['cr_layer', 'cr_pred', 'r_layer'],
)

model = build_cross_resistance_model(
    n_snps=X.shape[1],
    n_non_cr=len(non_cr_drugs),
    cr_pairs=cr_pairs,
    sparse_input=False,
    anchor_weights=anchor,
    anchor_strength=1e-5,
)
assert_output_order(model, model_drugs)
model.summary()

# --- Compile ----------------------------------------------------------
pos_weight_tf = compute_pos_weight(y_train)
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss=make_masked_weighted_bce(pos_weight_tf),
    metrics=[MaskedF1(), MaskedAUC()],
)

val_split = int(0.8 * len(x_train))
x_tr, x_val = x_train[:val_split], x_train[val_split:]
y_tr, y_val = y_train[:val_split], y_train[val_split:]

# NB: pass model_drugs, not label_drugs -- y is in model order here
per_drug_cb = PerDrugHistory(x_val, y_val, model_drugs)

history = model.fit(
    x_tr, y_tr,
    epochs=50,
    validation_data=(x_val, y_val),
    batch_size=64,
    callbacks=[
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=8,
                                      restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                          patience=3, min_lr=1e-6),
        per_drug_cb,
    ],
)

test_loss, test_f1, test_auc = model.evaluate(x_test, y_test, verbose=0)
print(f"Test loss: {test_loss:.4f}, Test F1: {
      test_f1:.4f}, Test AUC: {test_auc:.4f}")

# --- Thresholds (computed in model order) ------------------------------
y_pred_val = model.predict(x_val)

tb = pd.read_csv(METRICS / "naiive_perf_metrics.csv",
                 skiprows=1).rename(columns=str.lower)
tbsen_raw = [float(str(s).split('/')[0].strip('( '))
             for s in tb['sensitivity'].tolist()]
# match by drug name, not row position: comp.py writes rows in FINAL_SNP.json key order
tb_sens = dict(zip(tb['drug'].str.lower(), tbsen_raw))
missing = [d for d in label_drugs if d not in tb_sens]
if missing:
    raise ValueError(
        f"no tb-profiler sensitivity for {missing} in naiive_perf_metrics.csv")

best_thresholds = {}
for di, drug in enumerate(model_drugs):
    mask = ~np.isnan(y_val[:, di])
    thr, matched = pick_threshold(
        y_val[mask, di], y_pred_val[mask, di],
        target_sens=tb_sens[drug], target_spec=0.98,
    )
    best_thresholds[drug] = thr

# --- Test metrics ------------------------------------------------------
y_pred = model.predict(x_test)

rows, stats_by_drug = [], {}
for i, drug in enumerate(model_drugs):
    mask = ~np.isnan(y_test[:, i])
    yt, yp = y_test[mask, i], y_pred[mask, i]
    n = int(mask.sum())

    if n == 0 or len(np.unique(yt)) < 2:
        rows.append({'drug': drug, 'auc': None, 'sensitivity': None,
                     'specificity': None, 'f1': None, 'threshold': None, 'n': n})
        stats_by_drug[drug] = None
        continue

    auc_m = keras.metrics.AUC()
    auc_m.update_state(yt, yp)

    stats = confusion_stats(yt, yp, best_thresholds[drug])
    stats_by_drug[drug] = stats
    rows.append({'drug': drug, 'auc': round(float(auc_m.result().numpy()), 3),
                 'sensitivity': round(stats['sensitivity'], 3),
                 'specificity': round(stats['specificity'], 3),
                 'f1': round(stats['f1'], 3),
                 'threshold': round(best_thresholds[drug], 2), 'n': n})

# Report in the familiar alphabetical order
metrics_df = (pd.DataFrame(rows)
                .set_index('drug')
                .reindex(label_drugs)
                .reset_index())
metrics_df.to_csv(str(OUT / f'{name}_CR_by_drug_metrics.csv'), index=False)
print(metrics_df.to_string(index=False))

# Predictions back in alphabetical order
pred_alpha = reorder_columns(y_pred, model_drugs, label_drugs)
# best_thresholds is keyed by drug name, so column order doesn't matter here
save_prediction_tables(
    pred_alpha, label_drugs, best_thresholds,
    scores_path=OUT / f'{name}_CR_trained_predictions.csv',
    calls_path=OUT / f'{name}_CR_trained_calls.csv',
    run_ids=runs_test,
)

plot_confusion_grid(
    labels=label_drugs,
    stats_by_label=stats_by_drug,
    save_path_base=str(FIGURES / f'{name}_CR_confusion_matrices'),
    suptitle=f"{tname} CR Confusion matrices (S=Susceptible, R=Resistant)",
)

loss_df = pd.DataFrame({
    'epoch':      range(1, len(history.history['loss']) + 1),
    'train_loss': history.history['loss'],
    'val_loss':   history.history['val_loss'],
    'train_auc':  history.history['masked_auc'],
    'val_auc':    history.history['val_masked_auc'],
    'train_f1':   history.history['masked_f1'],
    'val_f1':     history.history['val_masked_f1'],
})
loss_df.to_csv(str(OUT / f'{name}_CR_loss_history.csv'), index=False)

plot_learning_curves(
    epochs=loss_df['epoch'],
    curves={
        "loss": {"train": loss_df['train_loss'], "val": loss_df['val_loss']},
        "auc":  {"train": loss_df['train_auc'], "val": loss_df['val_auc']},
        "f1":   {"train": loss_df['train_f1'], "val": loss_df['val_f1']},
    },
    save_path_base=str(FIGURES / f'{name}_CR_learning_curves'),
    title=f'Global learning curves ({tname} SNPs, cross-resistance)',
    test_values={"loss": test_loss, "auc": test_auc, "f1": test_f1},
)

# keep the saved model free of anchor kernels
strip_anchor_regularizers(model)
model.save(
    str(RESULTS / f'models/trained_models/{name}_CR_trained_model.keras'))
print(f"Saved: {name}_CR_trained_model.keras")
