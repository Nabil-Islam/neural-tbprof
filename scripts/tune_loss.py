"""
Grid search over the loss weights, by 5-fold CV on the training pool.
The test set is never loaded into the model or scored.

Grid (27 settings):
    alpha      : class weight on resistant isolates, (neg/pos)**alpha  {0.5, 0, -0.25}
    neg_weight : weight on susceptible isolates (FP cost)              {1, 2, 4}
    fp_gamma   : focus that weight on high-scoring susceptibles (FPs)  {0, 1, 2}
Setting 0 (alpha=0.5, neg_weight=1, fp_gamma=0) is the current loss.

For each setting, each CV fold is predicted by a model trained on the other
folds; the pooled out-of-fold predictions are scored per drug as
sensitivity at TB-Profiler's specificity and partial AUC (FPR <= 0.05).

Tune on `er` (default). The `filt` variants were selected using CV folds
(enrich_snps.py), so CV on `filt` would be optimistic. Use the chosen
setting for all three variant sets.

Usage (from repo root):
    # one setting per SLURM array task (0..26):
    python scripts/tune_loss.py er --setting $SLURM_ARRAY_TASK_ID --ref-runs <TN/FP json>
    # or all settings in one job:
    python scripts/tune_loss.py er --all --ref-runs <TN/FP json>
    # once all settings are done:
    python scripts/tune_loss.py er --summarise
    # epoch / learning-rate diagnostic on one fold, kept in tuning/er_diag/:
    python scripts/tune_loss.py er --setting 0 --folds 0 --max-epochs 300 --tag diag

Outputs in results/metrics/tuning/<set>/:
    setting_XX.csv       per-drug CV scores for one setting
    setting_XX_oof.npz   out-of-fold predictions + best epoch per fold
    summary.csv          settings ranked
    best_settings.json   chosen setting, read by train_new.py --tuned
"""

import argparse
import itertools
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from tbml.experiment import (DRUGS, BATCH_SIZE, MAX_EPOCHS, MONITOR, LEARNING_RATE,
                             load_variant_set, build_and_compile,
                             training_callbacks, per_drug_scores)
from tbml.reference import reference_specificity
from tbml.splits import load_split

ALPHAS = [0.5, 0.0, -0.25]
NEG_WEIGHTS = [1.0, 2.0, 4.0]
FP_GAMMAS = [0.0, 1.0, 2.0]
GRID = [dict(alpha=a, neg_weight=n, fp_gamma=g)
        for a, n, g in itertools.product(ALPHAS, NEG_WEIGHTS, FP_GAMMAS)]
BASELINE = 0  # alpha=0.5, neg_weight=1, fp_gamma=0: the current loss


def run_setting(idx, data, masks, target_spec, out, seed, max_epochs, learning_rate,
                only_folds=None):
    import tf_keras as keras

    setting = GRID[idx]
    X, y = data['X'], data['y']
    pool = masks['train_pool']
    fold = masks['fold']
    all_folds = sorted(int(k) for k in np.unique(fold[pool]))
    folds = all_folds if only_folds is None else [k for k in all_folds if k in only_folds]

    oof = np.full(y.shape, np.nan, dtype=np.float32)
    best_epochs = []
    curves = np.full((len(folds), 2, max_epochs), np.nan, dtype=np.float32)  # train/val loss

    for k in folds:
        tr = pool & (fold != k)
        va = pool & (fold == k)
        keras.utils.set_random_seed(seed + k)   # same per fold across settings
        model = build_and_compile(data['kernel'], y[tr], **setting,
                                  learning_rate=learning_rate)
        hist = model.fit(X[tr], y[tr], validation_data=(X[va], y[va]),
                         epochs=max_epochs, batch_size=BATCH_SIZE,
                         callbacks=training_callbacks(), verbose=2)
        best_epochs.append(int(np.argmin(hist.history[MONITOR]) + 1))
        fi = folds.index(k)
        n_ep = len(hist.history['loss'])
        curves[fi, 0, :n_ep] = hist.history['loss']
        curves[fi, 1, :n_ep] = hist.history['val_loss']
        oof[va] = model.predict(X[va], batch_size=4096, verbose=0)
        keras.backend.clear_session()
        print(f"setting {idx} {setting} fold {k}: best epoch {best_epochs[-1]}", flush=True)

    scored = pool & np.isin(fold, folds)
    scores = per_drug_scores(y[scored], oof[scored], DRUGS, target_spec)
    for key, v in setting.items():
        scores.insert(0, key, v)
    scores.insert(0, 'setting', idx)
    scores.to_csv(out / f'setting_{idx:02d}.csv', index=False)
    np.savez_compressed(out / f'setting_{idx:02d}_oof.npz', oof=oof[scored],
                        runs=data['runs'][scored], best_epochs=np.array(best_epochs),
                        folds=np.array(folds), curves=curves,
                        max_epochs=max_epochs, learning_rate=learning_rate)
    if max(best_epochs) >= max_epochs:
        print(f"WARNING: best epoch == max epochs ({max_epochs}) in some folds: "
              f"still improving when stopped; raise --max-epochs or --lr")
    print(scores[['drug', 'n_resistant', 'sens_at_spec', 'spec', 'pauc']].to_string(index=False))


def summarise(out):
    files = sorted(out.glob('setting_[0-9][0-9].csv'))
    if not files:
        raise SystemExit(f"no setting_XX.csv files in {out}")
    df = pd.concat([pd.read_csv(f) for f in files])
    done = sorted(df['setting'].unique())
    missing = [i for i in range(len(GRID)) if i not in done]
    if missing:
        print(f"WARNING: settings not finished: {missing}")

    agg = (df.groupby(['setting', 'alpha', 'neg_weight', 'fp_gamma'])
             .agg(mean_sens_at_spec=('sens_at_spec', 'mean'),
                  mean_pauc=('pauc', 'mean'), mean_auc=('auc', 'mean'))
             .reset_index())

    if BASELINE in done:
        base = df[df['setting'] == BASELINE].set_index('drug')['sens_at_spec']
        diff = df.assign(d=df['sens_at_spec'] - df['drug'].map(base))
        agg = agg.merge(
            diff.groupby('setting')['d'].agg(
                drugs_better=lambda s: int((s > 0.005).sum()),
                drugs_worse=lambda s: int((s < -0.005).sum())).reset_index(),
            on='setting')

    # rank: mean sensitivity at TB-Profiler's specificity, then partial AUC
    agg = agg.sort_values(['mean_sens_at_spec', 'mean_pauc'], ascending=False)
    agg.to_csv(out / 'summary.csv', index=False)

    # per-drug table, settings as columns, for checking what a setting trades
    wide = df.pivot_table(index='drug', columns='setting', values='sens_at_spec')
    wide.round(3).to_csv(out / 'sens_at_spec_by_drug.csv')

    capped = []
    for f in sorted(out.glob('setting_[0-9][0-9]_oof.npz')):
        z = np.load(f)
        cap = int(z['max_epochs']) if 'max_epochs' in z else MAX_EPOCHS
        if (z['best_epochs'] >= cap).any():
            capped.append(f.name[8:10])
    if capped:
        print(f"WARNING: settings {capped} were still improving at the epoch limit; "
              f"the ranking compares under-trained models")

    best = agg.iloc[0]
    oof = np.load(out / f"setting_{int(best['setting']):02d}_oof.npz")
    chosen = {
        'setting': int(best['setting']),
        'alpha': float(best['alpha']),
        'neg_weight': float(best['neg_weight']),
        'fp_gamma': float(best['fp_gamma']),
        'mean_sens_at_spec_cv': float(best['mean_sens_at_spec']),
        'mean_pauc_cv': float(best['mean_pauc']),
        'cv_best_epochs': oof['best_epochs'].tolist(),
        'max_epochs': int(oof['max_epochs']) if 'max_epochs' in oof else MAX_EPOCHS,
        'learning_rate': float(oof['learning_rate']) if 'learning_rate' in oof else LEARNING_RATE,
    }
    with open(out / 'best_settings.json', 'w') as f:
        json.dump(chosen, f, indent=2)

    with pd.option_context('display.width', 200):
        print(agg.round(4).to_string(index=False))
    print(f"\nchosen: {chosen}")
    print("Check sens_at_spec_by_drug.csv before accepting it: a small mean gain "
          "can hide a loss on one drug, and differences of ~0.01 on drugs with "
          "few resistant isolates (BDQ, CFZ, DLM, LZD) are within noise.")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('variant_set', choices=['who', 'er', 'filt'])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument('--setting', type=int, help=f'grid index 0..{len(GRID) - 1}')
    g.add_argument('--all', action='store_true', help='run every setting')
    g.add_argument('--summarise', action='store_true')
    ap.add_argument('--results', default='results')
    ap.add_argument('--ref-runs', default=None,
                    help="TB-Profiler TN/FP runs JSON (old_tbprof NEW_updated_runs.json); "
                         "target specificity is then computed on non-test isolates")
    ap.add_argument('--ref-csv', default=None,
                    help='fallback: metrics/naiive_perf_metrics.csv (all samples)')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--max-epochs', type=int, default=MAX_EPOCHS)
    ap.add_argument('--lr', type=float, default=LEARNING_RATE, help='Adam learning rate')
    ap.add_argument('--folds', type=int, nargs='+', default=None,
                    help='only these CV folds (for quick diagnostics; scores use those folds only)')
    ap.add_argument('--tag', default=None,
                    help='write to tuning/<set>_<tag>/ so runs with other epochs/lr are kept apart')
    args = ap.parse_args()

    results = Path(args.results)
    out = results / 'metrics/tuning' / (args.variant_set + (f'_{args.tag}' if args.tag else ''))
    out.mkdir(parents=True, exist_ok=True)

    if args.summarise:
        summarise(out)
        return

    if args.variant_set == 'filt':
        print("WARNING: tuning on filt; its variants were chosen using CV folds, "
              "so these CV scores are optimistic. Prefer `er`.")

    data = load_variant_set(results, args.variant_set)
    masks = load_split(results / 'essentials/splits.csv', data['runs'])
    print(f"train pool: {masks['train_pool'].sum()}  (test held out: {masks['test'].sum()})")

    ref_csv = args.ref_csv or (results / 'metrics/naiive_perf_metrics.csv')
    target_spec, src = reference_specificity(
        DRUGS, data['runs'][masks['train_pool']], runs_json=args.ref_runs,
        csv_path=None if args.ref_runs else ref_csv)
    print(f"target specificity from {src}:")
    print({d: round(v, 3) for d, v in target_spec.items()})
    with open(out / 'target_spec.json', 'w') as f:
        json.dump({'source': str(src), 'target_spec': target_spec}, f, indent=2)

    todo = range(len(GRID)) if args.all else [args.setting]
    for idx in todo:
        if not 0 <= idx < len(GRID):
            raise SystemExit(f"--setting must be in 0..{len(GRID) - 1}")
        if (out / f'setting_{idx:02d}.csv').exists():
            print(f"setting {idx} already done, skipping")
            continue
        run_setting(idx, data, masks, target_spec, out, args.seed,
                    args.max_epochs, args.lr, args.folds)


if __name__ == '__main__':
    main()
