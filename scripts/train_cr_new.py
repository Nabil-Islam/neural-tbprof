"""
Train the cross-resistance model under the same protocol as train_new.py,
so its results sit alongside the single-layer models in comp_new.py.

    python scripts/train_cr_new.py <who|er|filt> --tuned --tuned-set er_ep100 --ref-runs <TN/FP json>

Architecture (tbml.models.build_cross_resistance_model): drugs in a
cross-resistance pair (LFX-MFX, BDQ-CFZ, INH-ETO) also receive their
partner's learned cross-resistance signal; the other nine drugs use a
plain sigmoid layer. The model outputs [non-CR drugs, CR drugs], which is
not alphabetical: labels are reordered to model order for training and
predictions are reordered back to alphabetical for every output file.

Same as train_new.py:
  * fixed accession-level split from splits.csv; train on the CV folds
    except --val-fold, which is used for early stopping and thresholds;
    the test set is only scored
  * thresholds on the validation fold at TB-Profiler's specificity
  * loss weights from tune_loss.py (--tuned) or --alpha/--neg-weight/--fp-gamma
  * learning rate / epoch limit from the tuning run, early stopping on val loss
Starting weights come from the untrained CR model
(models/naiive_models/<name>_CR_naiive_model.h5), anchored as before.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import tf_keras as keras

from tbml.plotting import set_plot_style, plot_learning_curves, plot_confusion_grid
from tbml.losses import (make_masked_weighted_bce, compute_pos_weight,
                         masked_binary_crossentropy)
from tbml.metrics import MaskedF1, MaskedAUC
from tbml.confusion import confusion_stats
from tbml.callbacks import PerDrugHistory
from tbml.thresholds import threshold_at_spec
from tbml.predictions import save_prediction_tables
from tbml.models import build_cross_resistance_model, strip_anchor_regularizers
from tbml.weights import load_layer_weights
from tbml.ordering import (cross_resistance_output_order, reorder_columns,
                           assert_output_order)
from tbml.splits import load_split
from tbml.reference import reference_specificity
from tbml.experiment import (DRUGS, BATCH_SIZE, MAX_EPOCHS, LEARNING_RATE,
                             ANCHOR_STRENGTH, load_variant_set, training_callbacks)

CR_PAIRS = {
    "levofloxacin": "moxifloxacin",
    "moxifloxacin": "levofloxacin",
    "bedaquiline":  "clofazimine",
    "clofazimine":  "bedaquiline",
    "isoniazid":    "ethionamide",
    "ethionamide":  "isoniazid",
}

ap = argparse.ArgumentParser(description=__doc__,
                             formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument('variant_set', choices=['who', 'er', 'filt'])
ap.add_argument('--val-fold', type=int, default=0)
ap.add_argument('--tuned', action='store_true')
ap.add_argument('--tuned-set', default='er')
ap.add_argument('--alpha', type=float, default=0.5)
ap.add_argument('--neg-weight', type=float, default=1.0)
ap.add_argument('--fp-gamma', type=float, default=0.0)
ap.add_argument('--max-epochs', type=int, default=None)
ap.add_argument('--lr', type=float, default=None)
ap.add_argument('--ref-runs', default=None,
                help='TB-Profiler TN/FP runs JSON: target specificity on non-test isolates')
ap.add_argument('--results', default='results')
ap.add_argument('--seed', type=int, default=0)
args = ap.parse_args()

set_plot_style()
RESULTS = Path(args.results)
FIGURES = RESULTS / 'figures'
FIGURES.mkdir(parents=True, exist_ok=True)
METRICS = RESULTS / 'metrics'
OUT = METRICS / 'cr'
OUT.mkdir(parents=True, exist_ok=True)

# --- Loss weights / optimiser settings (as train_new.py) ---
if args.tuned:
    best_path = METRICS / 'tuning' / args.tuned_set / 'best_settings.json'
    with open(best_path) as f:
        best = json.load(f)
    loss_cfg = {k: best[k] for k in ('alpha', 'neg_weight', 'fp_gamma')}
    max_epochs = args.max_epochs or best.get('max_epochs', MAX_EPOCHS)
    lr = args.lr or best.get('learning_rate', LEARNING_RATE)
    print(f"settings from {best_path}")
else:
    loss_cfg = dict(alpha=args.alpha, neg_weight=args.neg_weight, fp_gamma=args.fp_gamma)
    max_epochs = args.max_epochs or MAX_EPOCHS
    lr = args.lr or LEARNING_RATE
print(f"loss {loss_cfg}, max epochs {max_epochs}, learning rate {lr}")

# --- Data: labels in alphabetical order, reordered to model order ---
data = load_variant_set(RESULTS, args.variant_set, load_kernel=False)
X, y_alpha, runs = data['X'], data['y'], data['runs']
name, tname = data['name'], data['tname']

non_cr_drugs = [d for d in DRUGS if d not in CR_PAIRS]
model_drugs = cross_resistance_output_order(non_cr_drugs, CR_PAIRS)
y = reorder_columns(y_alpha, DRUGS, model_drugs)
print("model output order:", model_drugs)

masks = load_split(RESULTS / 'essentials/splits.csv', runs, val_fold=args.val_fold)
x_tr, y_tr = X[masks['train']], y[masks['train']]
x_val, y_val = X[masks['val']], y[masks['val']]
x_test, y_test = X[masks['test']], y[masks['test']]
runs_test = runs[masks['test']]
print(f"train: {x_tr.shape}  val (fold {args.val_fold}): {x_val.shape}  test: {x_test.shape}")

target_spec, ref_src = reference_specificity(
    DRUGS, runs[masks['train_pool']], runs_json=args.ref_runs,
    csv_path=None if args.ref_runs else METRICS / 'naiive_perf_metrics.csv')
print(f"target specificity from {ref_src}")

# --- Model: warm start from the untrained CR model ---
anchor = load_layer_weights(
    RESULTS / f'models/naiive_models/{name}_CR_naiive_model.h5',
    ['cr_layer', 'cr_pred', 'r_layer'])

keras.utils.set_random_seed(args.seed)
model = build_cross_resistance_model(
    n_snps=X.shape[1], n_non_cr=len(non_cr_drugs), cr_pairs=CR_PAIRS,
    sparse_input=False, anchor_weights=anchor, anchor_strength=ANCHOR_STRENGTH)
assert_output_order(model, model_drugs)

pw = compute_pos_weight(y_tr, alpha=loss_cfg['alpha'])          # model order
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=lr),
    loss=make_masked_weighted_bce(pw, neg_weight=loss_cfg['neg_weight'],
                                  fp_gamma=loss_cfg['fp_gamma'], normalize='weight'),
    metrics=[masked_binary_crossentropy, MaskedF1(), MaskedAUC()])

per_drug_cb = PerDrugHistory(x_val, y_val, model_drugs)   # y is in model order
history = model.fit(x_tr, y_tr, epochs=max_epochs, validation_data=(x_val, y_val),
                    batch_size=BATCH_SIZE, callbacks=[*training_callbacks(), per_drug_cb])
if len(history.history['loss']) >= max_epochs:
    print(f"WARNING: training ran to the {max_epochs}-epoch limit without early stopping")

test_eval = model.evaluate(x_test, y_test, verbose=0, return_dict=True)
print(f"Test loss: {test_eval['loss']:.4f}, F1: {test_eval['masked_f1']:.4f}, "
      f"AUC: {test_eval['masked_auc']:.4f}")

# --- Thresholds at TB-Profiler's specificity, on the validation fold ---
y_pred_val = model.predict(x_val, verbose=0)
best_thresholds, val_spec = {}, {}
for di, drug in enumerate(model_drugs):
    m = ~np.isnan(y_val[:, di])
    thr = threshold_at_spec(y_val[m, di], y_pred_val[m, di], target_spec[drug])
    best_thresholds[drug] = thr
    val_spec[drug] = confusion_stats(y_val[m, di], y_pred_val[m, di], thr)['specificity']

# --- Test metrics (computed in model order, reported alphabetically) ---
y_pred = model.predict(x_test, verbose=0)
rows, stats_by_drug = [], {}
for i, drug in enumerate(model_drugs):
    m = ~np.isnan(y_test[:, i])
    yt, yp = y_test[m, i], y_pred[m, i]
    n = int(m.sum())
    if n == 0 or len(np.unique(yt)) < 2:
        rows.append({'drug': drug, 'n': n})
        stats_by_drug[drug] = None
        continue
    auc_m = keras.metrics.AUC()
    auc_m.update_state(yt, yp)
    st = confusion_stats(yt, yp, best_thresholds[drug])
    stats_by_drug[drug] = st
    rows.append({'drug': drug, 'auc': round(float(auc_m.result().numpy()), 3),
                 'sensitivity': round(st['sensitivity'], 3),
                 'specificity': round(st['specificity'], 3),
                 'f1': round(st['f1'], 3),
                 'threshold': round(best_thresholds[drug], 4),
                 'target_spec': round(target_spec[drug], 3),
                 'val_spec': round(float(val_spec[drug]), 3), 'n': n})

metrics_df = pd.DataFrame(rows).set_index('drug').reindex(DRUGS).reset_index()
metrics_df.to_csv(OUT / f'{name}_CR_by_drug_metrics.csv', index=False)
print(metrics_df.to_string(index=False))

pred_alpha = reorder_columns(y_pred, model_drugs, DRUGS)
save_prediction_tables(
    pred_alpha, DRUGS, best_thresholds,
    scores_path=OUT / f'{name}_CR_trained_predictions.csv',
    calls_path=OUT / f'{name}_CR_trained_calls.csv',
    run_ids=runs_test)

with open(OUT / f'{name}_CR_run_config.json', 'w') as f:
    json.dump({'loss': loss_cfg, 'val_fold': args.val_fold, 'seed': args.seed,
               'max_epochs': max_epochs, 'learning_rate': lr,
               'target_spec_source': str(ref_src), 'target_spec': target_spec,
               'thresholds': best_thresholds,
               'epochs_run': len(history.history['loss'])}, f, indent=2)

plot_confusion_grid(
    labels=DRUGS, stats_by_label=stats_by_drug,
    save_path_base=str(FIGURES / f'{name}_CR_confusion_matrices'),
    suptitle=f"{tname} CR Confusion matrices (S=Susceptible, R=Resistant)")

h = history.history
loss_df = pd.DataFrame({'epoch': range(1, len(h['loss']) + 1),
                        'train_loss': h['loss'], 'val_loss': h['val_loss'],
                        'train_auc': h['masked_auc'], 'val_auc': h['val_masked_auc'],
                        'train_f1': h['masked_f1'], 'val_f1': h['val_masked_f1']})
loss_df.to_csv(OUT / f'{name}_CR_loss_history.csv', index=False)
plot_learning_curves(
    epochs=loss_df['epoch'],
    curves={"loss": {"train": loss_df['train_loss'], "val": loss_df['val_loss']},
            "auc": {"train": loss_df['train_auc'], "val": loss_df['val_auc']},
            "f1": {"train": loss_df['train_f1'], "val": loss_df['val_f1']}},
    save_path_base=str(FIGURES / f'{name}_CR_learning_curves'),
    title=f'Global learning curves ({tname} SNPs, cross-resistance)',
    test_values={"loss": test_eval['loss'], "auc": test_eval['masked_auc'],
                 "f1": test_eval['masked_f1']})

strip_anchor_regularizers(model)
(RESULTS / 'models/trained_models').mkdir(parents=True, exist_ok=True)
model.save(str(RESULTS / f'models/trained_models/{name}_CR_trained_model.keras'))
print(f"Saved: {name}_CR_trained_model.keras")
