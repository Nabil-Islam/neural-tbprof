"""
Train the anchored single-layer model on one variant set and evaluate it
once on the held-out test set.

    python scripts/train_new.py <who|er|filt> --tuned --ref-runs <TN/FP json>
    python scripts/train_new.py er --alpha 0 --neg-weight 2 --fp-gamma 1

Data split: results/essentials/splits.csv (make_splits.py). The model is
trained on the CV folds except --val-fold, which is used for early stopping,
for choosing thresholds, and (filt only) for selecting SNPs in the ablation.
The test set is only scored.

Loss weights: --tuned reads results/metrics/tuning/<tuned-set>/best_settings.json
written by tune_loss.py --summarise; otherwise --alpha/--neg-weight/--fp-gamma
(defaults = the original loss).

Thresholds: per drug, the threshold on the validation fold that gives
TB-Profiler's specificity, so test sensitivity can be compared with
TB-Profiler's like-for-like.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
import tf_keras as keras
from tqdm import tqdm

from tbml.plotting import set_plot_style, plot_learning_curves, plot_confusion_grid
from tbml.metrics import MaskedF1, MaskedAUC
from tbml.confusion import confusion_stats
from tbml.callbacks import PerDrugHistory
from tbml.thresholds import threshold_at_spec
from tbml.predictions import save_prediction_tables
from tbml.models import strip_anchor_regularizers
from tbml.splits import load_split
from tbml.reference import reference_specificity
from tbml.experiment import (DRUGS, BATCH_SIZE, MAX_EPOCHS, LEARNING_RATE, load_variant_set,
                             build_and_compile, training_callbacks)

OLD_TBPROF = '/data/scratch/projects/punim1703/false_negative/old_tbprof/results'

ap = argparse.ArgumentParser(description=__doc__,
                             formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument('variant_set', choices=['who', 'er', 'filt'])
ap.add_argument('--val-fold', type=int, default=0,
                help='CV fold used as validation (keep equal to enrich_snps.py --exclude-fold)')
ap.add_argument('--tuned', action='store_true',
                help='use the loss weights chosen by tune_loss.py')
ap.add_argument('--tuned-set', default='er',
                help='variant set the tuning was run on (default er)')
ap.add_argument('--alpha', type=float, default=0.5)
ap.add_argument('--neg-weight', type=float, default=1.0)
ap.add_argument('--fp-gamma', type=float, default=0.0)
ap.add_argument('--ref-runs', default=None,
                help='TB-Profiler TN/FP runs JSON (NEW_updated_runs.json): target '
                     'specificity computed on non-test isolates')
ap.add_argument('--results', default='results')
ap.add_argument('--filtered-dir', default=f'{OLD_TBPROF}/matrices',
                help='where filtered_<drug>_snps.csv live (filt ablation)')
ap.add_argument('--flagged', default=f'{OLD_TBPROF}/flagged_snps.csv')
ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--max-epochs', type=int, default=None,
                help='default: from best_settings.json with --tuned, else MAX_EPOCHS')
ap.add_argument('--lr', type=float, default=None,
                help='Adam learning rate (default: from best_settings.json with --tuned)')
args = ap.parse_args()

set_plot_style()
arg = args.variant_set
drugs = DRUGS

# --- Paths ---
RESULTS = Path(args.results)
FIGURES = RESULTS / 'figures'
FIGURES.mkdir(parents=True, exist_ok=True)
METRICS = RESULTS / 'metrics'
METRICS.mkdir(parents=True, exist_ok=True)
OUT = METRICS / arg        # per-run outputs; naiive_perf_metrics.csv stays in METRICS
OUT.mkdir(parents=True, exist_ok=True)
FILTERED = Path(args.filtered_dir)
FLAGGED_CSV = Path(args.flagged)

# --- Loss weights ---
if args.tuned:
    best_path = METRICS / 'tuning' / args.tuned_set / 'best_settings.json'
    with open(best_path) as f:
        best = json.load(f)
    loss_cfg = {k: best[k] for k in ('alpha', 'neg_weight', 'fp_gamma')}
    print(f"loss weights from {best_path}: {loss_cfg}")
    max_epochs = args.max_epochs or best.get('max_epochs', MAX_EPOCHS)
    lr = args.lr or best.get('learning_rate', LEARNING_RATE)
else:
    loss_cfg = dict(alpha=args.alpha, neg_weight=args.neg_weight, fp_gamma=args.fp_gamma)
    print(f"loss weights from command line: {loss_cfg}")
    max_epochs = args.max_epochs or MAX_EPOCHS
    lr = args.lr or LEARNING_RATE
print(f"max epochs {max_epochs}, learning rate {lr}")

# --- Load data ---
data = load_variant_set(RESULTS, arg)
X, y, runs, kernel = data['X'], data['y'], data['runs'], data['kernel']
name, tname = data['name'], data['tname']

# --- Fixed accession-level split from make_splits.py ---
masks = load_split(RESULTS / 'essentials/splits.csv', runs, val_fold=args.val_fold)
x_tr, y_tr = X[masks['train']], y[masks['train']]
x_val, y_val = X[masks['val']], y[masks['val']]
x_test, y_test = X[masks['test']], y[masks['test']]
runs_test = runs[masks['test']]
print(f"train: {x_tr.shape}  val (fold {args.val_fold}): {x_val.shape}  test: {x_test.shape}")

# --- TB-Profiler specificity to match (never computed on test isolates
#     when --ref-runs is given) ---
target_spec, ref_src = reference_specificity(
    drugs, runs[masks['train_pool']], runs_json=args.ref_runs,
    csv_path=None if args.ref_runs else METRICS / 'naiive_perf_metrics.csv')
print(f"target specificity from {ref_src}")

# --- Build, compile, train ---
keras.utils.set_random_seed(args.seed)
model = build_and_compile(kernel, y_tr, **loss_cfg, learning_rate=lr,
                          extra_metrics=[MaskedF1(), MaskedAUC()])

per_drug_cb = PerDrugHistory(x_val, y_val, drugs)
history = model.fit(
    x_tr, y_tr,
    epochs=max_epochs,
    validation_data=(x_val, y_val),
    batch_size=BATCH_SIZE,
    callbacks=[*training_callbacks(), per_drug_cb],
)

# --- Evaluate on test set ---
test_eval = model.evaluate(x_test, y_test, verbose=0, return_dict=True)
test_loss, test_f1, test_auc = test_eval['loss'], test_eval['masked_f1'], test_eval['masked_auc']
print(f"Test loss: {test_loss:.4f}, Test F1: {test_f1:.4f}, Test AUC: {test_auc:.4f}")

# --- Thresholds: TB-Profiler's specificity, set on the validation fold ---
y_pred_val = model.predict(x_val, verbose=0)
best_thresholds, val_spec = {}, {}
for di, drug in enumerate(drugs):
    mask = ~np.isnan(y_val[:, di])
    thr = threshold_at_spec(y_val[mask, di], y_pred_val[mask, di], target_spec[drug])
    best_thresholds[drug] = thr
    val_spec[drug] = confusion_stats(y_val[mask, di], y_pred_val[mask, di], thr)['specificity']

with open(OUT / f'{name}_run_config.json', 'w') as f:
    json.dump({'loss': loss_cfg, 'val_fold': args.val_fold, 'seed': args.seed,
               'target_spec_source': str(ref_src), 'target_spec': target_spec,
               'thresholds': best_thresholds,
               'max_epochs': max_epochs, 'learning_rate': lr,
               'epochs_run': len(history.history['loss'])}, f, indent=2)
if len(history.history['loss']) >= max_epochs:
    print(f"WARNING: training ran to the {max_epochs}-epoch limit without early stopping")

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
y_pred = model.predict(x_test, verbose=0)

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
                 'threshold': round(thresh, 4),
                 'target_spec': round(target_spec[drug], 3),
                 'val_spec': round(float(val_spec[drug]), 3), 'n': n})

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

    flagged = set(pd.read_csv(FLAGGED_CSV)['snp'])
    print(f"{len(flagged)} flagged SNPs excluded from ablation")

    final_kernel, final_bias = model.get_layer('large_layer').get_weights()

    good_snps = {d: {} for d in drugs}
    SENS_THR = 0.005

    # SNPs are *selected* on the validation fold, never on test: choosing
    # which weights to drop by test sensitivity and then reporting test
    # sensitivity would leak. Test is only used to evaluate the result below.
    y_pred_val_base = model.predict(x_val, verbose=0)
    val_base_sens = {}
    for d in drugs:
        di = drugs.index(d)
        m = ~np.isnan(y_val[:, di])
        val_base_sens[d] = confusion_stats(
            y_val[m, di], y_pred_val_base[m, di], best_thresholds[d])['sensitivity']

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
            if snp not in snp_lu:          # (index 0 is a valid SNP)
                continue

            W_ablated = final_kernel.copy()
            W_ablated[snp_lu[snp], di] = 0.0
            model.get_layer('large_layer').set_weights([W_ablated, final_bias])
            y_pred_val_ablated = model.predict(x_val, verbose=0)

            mask = ~np.isnan(y_val[:, di])
            yt = y_val[mask, di]
            yp = y_pred_val_ablated[mask, di]

            stats = confusion_stats(yt, yp, best_thresholds[d])
            diff = stats['sensitivity'] - val_base_sens[d]

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
    back_select_pred = model.predict(x_test, verbose=0)

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
                   'threshold': round(best_thresholds[drug], 4), 'n': n})

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
