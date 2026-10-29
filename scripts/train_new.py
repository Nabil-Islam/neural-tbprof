import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import h5py
import tensorflow as tf
import tf_keras as keras
from tqdm import tqdm

from tbml.plotting import set_plot_style, plot_learning_curves, plot_confusion_grid
from tbml.losses import make_masked_weighted_bce, compute_pos_weight
from tbml.metrics import MaskedF1, MaskedAUC
from tbml.confusion import confusion_stats
from tbml.callbacks import PerDrugHistory
from tbml.thresholds import pick_threshold
from tbml.predictions import save_prediction_tables
from tbml.models import build_anchored_dense_model, strip_anchor_regularizers

set_plot_style()

# --- Run config from CLI arg ---
arg = sys.argv[1]
NAME_MAP = {
    'who': ('who', 'WHO'),
    'er': ('expert_rules', 'Expert rules'),
    'filt': ('filtered_expert_rules', 'Filtered expert rules'),
}
if arg not in NAME_MAP:
    raise ValueError(f"Unknown arg '{arg}', expected one of {list(NAME_MAP)}")
name, tname = NAME_MAP[arg]

# --- Paths ---
SCRIPTS = Path.cwd()
RESULTS = SCRIPTS / 'results'
FIGURES = RESULTS / 'figures'
FIGURES.mkdir(parents=True, exist_ok=True)
METRICS = RESULTS / 'metrics'
METRICS.mkdir(parents=True, exist_ok=True)
OUT = METRICS / arg        # per-run outputs; naiive_perf_metrics.csv stays in METRICS
OUT.mkdir(parents=True, exist_ok=True)
FILTERED = Path(
    '/data/scratch/projects/punim1703/false_negative/old_tbprof/results/matrices')
FLAGGED = Path(
    '/data/scratch/projects/punim1703/false_negative/old_tbprof/results')

drugs = ['amikacin', 'bedaquiline', 'capreomycin', 'clofazimine', 'delamanid',
         'ethambutol', 'ethionamide', 'isoniazid', 'kanamycin', 'levofloxacin',
         'linezolid', 'moxifloxacin', 'pyrazinamide', 'rifampicin', 'streptomycin']

# --- Load feature + label matrices ---
matrix = sp.load_npz(RESULTS / f'matrices/{name}/{name}_AUG_matrix.npz')
X = matrix.toarray().astype(np.float32)

label_matrix = pd.read_csv(RESULTS / 'essentials/labelled_vcfs.csv')
# keep sample ids before dropping
runs = label_matrix['run'].to_numpy()
label_matrix = label_matrix.drop(columns=['run'])
y = label_matrix.to_numpy().astype(np.float32)

# --- 80/20 split ---
rng = np.random.default_rng(42)
idx = rng.permutation(len(X))
split = int(0.8 * len(X))

x_train, x_test = X[idx[:split]], X[idx[split:]]
y_train, y_test = y[idx[:split]], y[idx[split:]]
runs_test = runs[idx[split:]]                  # same shuffle as x_test/y_test

print(f"x_train: {x_train.shape}, y_train: {y_train.shape}")
print(f"x_test:  {x_test.shape},  y_test:  {y_test.shape}")

# --- Load pretrained weights ---
with h5py.File(str(RESULTS / f'models/naiive_models/{name}_naiive_model.h5'), 'r') as f:
    kernel_init = f['model_weights/large_layer/large_layer/kernel'][:]
    # bias = f['model_weights/large_layer/large_layer/bias'][:]

# -- experiment: scale up kernel, reset bias --
kernel = kernel_init * 10.0
p = np.nanmean(y_train, axis=0)
bias = np.log(p / (1-p))

# --- Build anchored model ---
model = build_anchored_dense_model(
    input_dim=X.shape[1], n_outputs=15,
    kernel_init=kernel, bias_init=bias,
    layer_name='large_layer', activation='sigmoid', anchor_strength=1e-5,
)

# --- Compile with masked, class-weighted loss ---
pos_weight_tf = compute_pos_weight(y_train)
masked_weighted_bce = make_masked_weighted_bce(pos_weight_tf)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss=masked_weighted_bce,
    metrics=[MaskedF1(), MaskedAUC()]
)

# --- Train/val split (manual, for per-drug callback) ---
val_split = int(0.8 * len(x_train))
x_tr, x_val = x_train[:val_split], x_train[val_split:]
y_tr, y_val = y_train[:val_split], y_train[val_split:]

per_drug_cb = PerDrugHistory(x_val, y_val, drugs)

early_stop = keras.callbacks.EarlyStopping(
    monitor='val_loss', patience=8, restore_best_weights=True)
reduce_lr = keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6)

# --- Train ---
history = model.fit(
    x_tr, y_tr,
    epochs=50,
    validation_data=(x_val, y_val),
    batch_size=64,
    callbacks=[early_stop, reduce_lr, per_drug_cb]
)

# --- Evaluate on test set ---
test_loss, test_f1, test_auc = model.evaluate(x_test, y_test, verbose=0)
print(f"Test loss: {test_loss:.4f}, Test F1: {
      test_f1:.4f}, Test AUC: {test_auc:.4f}")

y_pred_val = model.predict(x_val)

# --- Reference sensitivities from tb-profiler as a floor ---
tb = pd.read_csv(METRICS / "naiive_perf_metrics.csv",
                 skiprows=1).rename(columns=str.lower)
tbsen_raw = [float(str(s).split('/')[0].strip('( '))
             for s in tb['sensitivity'].tolist()]
# match by drug name, not row position: comp.py writes rows in FINAL_SNP.json key order
tb_sens = dict(zip(tb['drug'].str.lower(), tbsen_raw))
missing = [d for d in drugs if d not in tb_sens]
if missing:
    raise ValueError(
        f"no tb-profiler sensitivity for {missing} in naiive_perf_metrics.csv")

# --- Pick per-drug thresholds ---
best_thresholds = {}
for di, drug in enumerate(drugs):
    mask = ~np.isnan(y_val[:, di])
    thr, matched = pick_threshold(
        y_val[mask, di], y_pred_val[mask, di],
        target_sens=tb_sens[drug], target_spec=0.98,
    )
    best_thresholds[drug] = thr

# --- Save global loss history ---
loss_df = pd.DataFrame({
    'epoch':      range(1, len(history.history['loss']) + 1),
    'train_loss': history.history['loss'],
    'val_loss':   history.history['val_loss'],
    'train_auc':  history.history['masked_auc'],
    'val_auc':    history.history['val_masked_auc'],
    'train_f1':   history.history['masked_f1'],
    'val_f1':     history.history['val_masked_f1'],
})
loss_df['test_loss'] = test_loss
loss_df['test_auc'] = test_auc
loss_df.to_csv(str(OUT / f'{name}_loss_history.csv'), index=False)
print(f"Saved: {name}_loss_history.csv")

# --- Global learning curves (loss, AUC, F1) ---
plot_learning_curves(
    epochs=loss_df['epoch'],
    curves={
        "loss": {"train": loss_df['train_loss'], "val": loss_df['val_loss']},
        "auc":  {"train": loss_df['train_auc'], "val": loss_df['val_auc']},
        "f1":   {"train": loss_df['train_f1'], "val": loss_df['val_f1']},
    },
    save_path_base=str(FIGURES / f'{name}_learning_curves'),
    title=f'Global learning curves ({tname} SNPs)',
    test_values={"loss": test_loss, "auc": test_auc, "f1": test_f1},
)
print(f"Saved: {name}_learning_curves.png")

# --- Per-drug learning curves ---
for drug in drugs:
    drug_epochs = range(1, len(per_drug_cb.history[drug]['loss']) + 1)
    plot_learning_curves(
        epochs=drug_epochs,
        curves={
            "loss": {"val": per_drug_cb.history[drug]['loss']},
            "auc":  {"val": per_drug_cb.history[drug]['auc']},
            "f1":   {"val": per_drug_cb.history[drug]['f1']},
        },
        save_path_base=str(FIGURES / f'{name}_{drug}_learning_curves'),
        title=f'{drug.capitalize()} ({tname} SNPs)',
        figsize=(14, 4),
    )
    print(f"Saved: {name}_{drug}_learning_curves.png")

# --- Save per-drug history to CSV ---
per_drug_rows = []
for drug in drugs:
    for ep, (l, a, f) in enumerate(zip(
            per_drug_cb.history[drug]['loss'],
            per_drug_cb.history[drug]['auc'],
            per_drug_cb.history[drug]['f1']), start=1):
        per_drug_rows.append({'drug': drug.capitalize(), 'epoch': ep, 'val_loss': l,
                              'val_auc': a, 'val_f1': f})
pd.DataFrame(per_drug_rows).to_csv(
    str(OUT / f'{name}_per_drug_history.csv'), index=False)
print(f"Saved: {name}_per_drug_history.csv")

# --- Per-drug metrics + confusion matrices ---
y_pred = model.predict(x_test)

save_prediction_tables(
    y_pred, drugs, best_thresholds,
    scores_path=OUT / f'{name}_trained_predictions.csv',
    calls_path=OUT / f'{name}_trained_calls.csv',
    run_ids=runs_test,
)
print(f"Saved: {name}_trained_predictions.csv, {name}_trained_calls.csv")

rows = []
stats_by_drug = {}
for i, drug in enumerate(drugs):
    mask = ~np.isnan(y_test[:, i])
    yt = y_test[mask, i]
    yp = y_pred[mask, i]
    n = mask.sum()

    if n == 0 or len(np.unique(yt)) < 2:
        rows.append({'drug': drug, 'auc': None, 'sensitivity': None,
                     'specificity': None, 'f1': None, 'threshold': None, 'n': n})
        stats_by_drug[drug] = None
        continue

    auc_m = tf.keras.metrics.AUC()
    auc_m.update_state(yt, yp)
    auc_val = auc_m.result().numpy()

    thresh = best_thresholds[drug]
    stats = confusion_stats(yt, yp, thresh)
    stats_by_drug[drug] = stats

    rows.append({'drug': drug, 'auc': round(float(auc_val), 3),
                 'sensitivity': round(stats['sensitivity'], 3),
                 'specificity': round(stats['specificity'], 3),
                 'f1': round(stats['f1'], 3),
                 'threshold': round(thresh, 2), 'n': n})

metrics_df = pd.DataFrame(rows)

plot_confusion_grid(
    labels=drugs,
    stats_by_label=stats_by_drug,
    save_path_base=str(FIGURES / f'{name}_confusion_matrices'),
    suptitle=f"{tname} Confusion matrices (S=Susceptible, R=Resistant)",
    ncols=5, figsize=(18, 10),
)
print(f"Saved: {name}_confusion_matrices.png")

# --- SNP ablation / back-selection (filtered_expert_rules only) ---
if name == 'filtered_expert_rules':
    with open(RESULTS / 'matrices/filtered_expert_rules/filtered_expert_rules_LOOKUP_SNP.json', 'r') as f:
        snp_lu = json.load(f)

    flagged = pd.read_csv(FLAGGED / 'flagged_snps.csv')['snp'].tolist()
    print(flagged)

    final_kernel, final_bias = model.get_layer('large_layer').get_weights()

    good_snps = {d: {} for d in drugs}
    SENS_THR = 0.005

    for d in tqdm(drugs, total=len(drugs), desc='sens checking'):
        fpath = FILTERED / f'{d}/filtered_{d}_snps.csv'
        if not fpath.exists():
            print(f'WARNING: skipping {d}, file missing')
            continue

        try:
            fsnps_df = pd.read_csv(fpath)
        except pd.errors.EmptyDataError:
            print(f'WARNING: skipping {d}, file is empty: {fpath}')
            continue

        if 'snp' not in fsnps_df.columns or fsnps_df.empty:
            print(f'WARNING: skipping {d}, no SNPs found')
            continue

        fsnps = fsnps_df['snp'].tolist()
        di = drugs.index(d)

        for snp in fsnps:
            if not snp_lu.get(snp):
                continue

            W_ablated = final_kernel.copy()
            W_ablated[snp_lu[snp], di] = 0.0
            model.get_layer('large_layer').set_weights([W_ablated, final_bias])
            y_pred_test_ablated = model.predict(x_test)

            mask = ~np.isnan(y_test[:, di])
            yt = y_test[mask, di]
            yp = y_pred_test_ablated[mask, di]

            stats = confusion_stats(yt, yp, best_thresholds[d])
            baseline_sens = metrics_df.loc[metrics_df['drug']
                                           == d, 'sensitivity'].item()
            diff = stats['sensitivity'] - baseline_sens

            if abs(diff) >= SENS_THR and snp not in flagged:
                good_snps[d][snp] = float(diff)

    with open(RESULTS / 'essentials/sens_diff_snps.json', 'w') as f:
        json.dump(good_snps, f)

    # --- Back-selection: drop SNPs that only helped, check metrics ---
    back_select_weight = final_kernel.copy()
    for d, snps in good_snps.items():
        di = drugs.index(d)
        for snp, val in snps.items():
            if float(val) >= 0:
                back_select_weight[snp_lu[snp], di] = 0.0

    model.get_layer('large_layer').set_weights(
        [back_select_weight, final_bias])
    back_select_pred = model.predict(x_test)

    r2 = []
    for drug in drugs:
        di = drugs.index(drug)
        mask = ~np.isnan(y_test[:, di])
        yt = y_test[mask, di]
        yp = back_select_pred[mask, di]
        n = int(mask.sum())

        if n == 0 or len(np.unique(yt)) < 2:
            r2.append({'drug': drug, 'auc': None, 'sensitivity': None,
                       'specificity': None, 'f1': None, 'threshold': None, 'n': n})
            continue

        auc_m = tf.keras.metrics.AUC()
        auc_m.update_state(yt, yp)
        auc_val = auc_m.result().numpy()

        stats = confusion_stats(yt, yp, best_thresholds[drug])
        r2.append({'drug': drug, 'auc': round(float(auc_val), 3),
                   'sensitivity': round(stats['sensitivity'], 3),
                   'specificity': round(stats['specificity'], 3),
                   'f1': round(stats['f1'], 3),
                   'threshold': round(best_thresholds[drug], 2), 'n': n})

    metrics_df_2 = pd.DataFrame(r2)
    metrics_df_2.to_csv(
        str(OUT / f'{name}_back_selected_by_drug_metrics.csv'), index=False)
    print(f"Saved: {name}_back_selected_by_drug_metrics.csv")
    print(metrics_df_2.to_string(index=False))

    # Restore weights for saving
    model.get_layer('large_layer').set_weights([final_kernel, final_bias])

metrics_df.to_csv(str(OUT / f'{name}_by_drug_metrics.csv'), index=False)
print(f"Saved: {name}_by_drug_metrics.csv")
print(metrics_df.to_string(index=False))

# --- Save model ---
# keep the saved model free of anchor kernels
strip_anchor_regularizers(model)
model.save(str(RESULTS / f'models/trained_models/{name}_trained_model.keras'))
model.export(
    str(RESULTS / f'models/trained_models/{name}_trained_saved_model'))
print(f"Saved: {name}_trained_model.keras")
