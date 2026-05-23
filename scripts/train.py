from sklearn.metrics import roc_curve
import sys
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt
import tensorflow as tf
import tf_keras as keras
import numpy as np
import h5py
import pandas as pd
import scipy.sparse as sp
import matplotlib
import json
import seaborn as sns
matplotlib.use('Agg')  # headless backend for HPC

sns.set_palette("muted")
plt.rcParams.update({
    "font.family":       "sans-serif",
    "font.sans-serif":   ["Nimbus Sans", "Liberation Sans", "DejaVu Sans"],
    "font.size":         8,
    "axes.linewidth":    0.75,
    "xtick.major.width": 0.75,
    "ytick.major.width": 0.75,
    "xtick.major.size":  3.5,
    "ytick.major.size":  3.5,
    "xtick.direction":   "out",
    "ytick.direction":   "out",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "pdf.fonttype":      42,
    "svg.fonttype":      "none",
})


arg = sys.argv[1]

if arg == 'who':
    name = 'who'
    tname = 'WHO'
if arg == 'er':
    name = 'expert_rules'
    tname = name.replace("_", " ").capitalize()
if arg == 'filt':
    name = 'filtered_expert_rules'
    tname = name.replace("_", " ").capitalize()

# --- Paths ---
SCRIPTS = Path.cwd()
RESULTS = SCRIPTS / 'results'
FIGURES = RESULTS / 'figures'
FIGURES.mkdir(parents=True, exist_ok=True)
METRICS = RESULTS / 'metrics'
METRICS.mkdir(parents=True, exist_ok=True)
FILTERED = Path(
    '/data/scratch/projects/punim1703/false_negative/old_tbprof/results/matrices')

drugs = ['amikacin', 'bedaquiline', 'capreomycin', 'clofazimine', 'delamanid',
         'ethambutol', 'ethionamide', 'isoniazid', 'kanamycin', 'levofloxacin',
         'linezolid', 'moxifloxacin', 'pyrazinamide', 'rifampicin', 'streptomycin']

# --- Load feature matrix ---
matrix = sp.load_npz(RESULTS / f'matrices/{name}/{name}_AUG_matrix.npz')
X = matrix.toarray().astype(np.float32)

# --- Load label matrix ---
label_matrix = pd.read_csv(RESULTS / 'essentials/labelled_vcfs.csv')
label_matrix = label_matrix.drop(columns=['run'])
y = label_matrix.to_numpy().astype(np.float32)

# --- 80/20 split ---
rng = np.random.default_rng(42)
idx = rng.permutation(len(X))
split = int(0.8 * len(X))

x_train, x_test = X[idx[:split]], X[idx[split:]]
y_train, y_test = y[idx[:split]], y[idx[split:]]

print(f"x_train: {x_train.shape}, y_train: {y_train.shape}")
print(f"x_test:  {x_test.shape},  y_test:  {y_test.shape}")

# --- Load weights ---
with h5py.File(str(RESULTS / f'models/naiive_models/{name}_naiive_model.h5'), 'r') as f:
    kernel_init = f['model_weights/large_layer/large_layer/kernel'][:]
    bias = f['model_weights/large_layer/large_layer/bias'][:]

# -- experiment---
kernel = kernel_init*10.0
bias = np.full((15,), -5.0).astype(np.float32)

w_init = tf.constant(kernel, dtype=tf.float32)


def anchor_reg(w):
    return 1e-5*tf.reduce_sum(tf.square(w-w_init))


# --- Build model ---
inputs = keras.Input(shape=(X.shape[1],))
outputs = keras.layers.Dense(
    15, activation='sigmoid', name='large_layer', kernel_regularizer=anchor_reg)(inputs)
model = keras.Model(inputs=inputs, outputs=outputs)
model.get_layer('large_layer').set_weights([kernel, bias])


# --- Masked loss ---
def masked_binary_crossentropy(y_true, y_pred):
    mask = tf.cast(~tf.math.is_nan(y_true), tf.float32)
    y_true_clean = tf.where(tf.math.is_nan(
        y_true), tf.zeros_like(y_true), y_true)
    loss = tf.keras.backend.binary_crossentropy(y_true_clean, y_pred)
    loss = loss * mask
    return tf.reduce_sum(loss) / (tf.reduce_sum(mask) + 1e-8)


# --Masked weighted BCE --
pos_counts = np.nansum(y_train, axis=0)
neg_counts = np.nansum(1 - y_train, axis=0)
pos_weight = neg_counts / (pos_counts + 1e-8)
pos_weight = np.sqrt(neg_counts / (pos_counts + 1e-8))
pos_weight_tf = tf.constant(pos_weight, dtype=tf.float32)


def masked_weighted_bce(y_true, y_pred):
    mask = tf.cast(~tf.math.is_nan(y_true), tf.float32)
    y_true_clean = tf.where(tf.math.is_nan(
        y_true), tf.zeros_like(y_true), y_true)
    weights = y_true_clean * pos_weight_tf + (1 - y_true_clean)
    loss = tf.keras.backend.binary_crossentropy(
        y_true_clean, y_pred) * weights * mask
    return tf.reduce_sum(loss) / (tf.reduce_sum(mask) + 1e-8)


# --- Masked F1 metric ---
class MaskedF1(keras.metrics.Metric):
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


# --- Masked AUC metric ---
class MaskedAUC(keras.metrics.Metric):
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


# --- Per-drug history callback ---
class PerDrugHistory(keras.callbacks.Callback):
    def __init__(self, x_val, y_val, drugs):
        super().__init__()
        self.x_val = x_val
        self.y_val = y_val
        self.drugs = drugs
        self.history = {drug: {'loss': [], 'auc': [], 'f1': []}
                        for drug in drugs}
        self._auc_metric = keras.metrics.AUC()

    def on_epoch_end(self, epoch, logs=None):
        y_pred = self.model.predict(self.x_val, verbose=0)
        for i, drug in enumerate(self.drugs):
            mask = ~np.isnan(self.y_val[:, i])
            yt = self.y_val[mask, i]
            yp = y_pred[mask, i]

            if len(yt) == 0 or len(np.unique(yt)) < 2:
                self.history[drug]['loss'].append(np.nan)
                self.history[drug]['auc'].append(np.nan)
                self.history[drug]['f1'].append(np.nan)
                continue

            # loss
            loss = -np.mean(yt * np.log(yp + 1e-8) +
                            (1 - yt) * np.log(1 - yp + 1e-8))

            # auc
            self._auc_metric.reset_state()
            self._auc_metric.update_state(yt, yp)
            auc_val = self._auc_metric.result().numpy()

            # f1
            yb = (yp > 0.5).astype(float)
            tp = ((yb == 1) & (yt == 1)).sum()
            fp = ((yb == 1) & (yt == 0)).sum()
            fn = ((yb == 0) & (yt == 1)).sum()
            prec = tp / (tp + fp + 1e-8)
            rec = tp / (tp + fn + 1e-8)
            f1 = 2 * prec * rec / (prec + rec + 1e-8)

            self.history[drug]['loss'].append(loss)
            self.history[drug]['auc'].append(auc_val)
            self.history[drug]['f1'].append(f1)


# --- Compile ---
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

# --- Callbacks ---
early_stop = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=8,
    restore_best_weights=True
)
reduce_lr = keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,
    patience=3,
    min_lr=1e-6
)

# --- Train ---
history = model.fit(
    x_tr, y_tr,
    epochs=50,
    validation_data=(x_val, y_val),
    batch_size=64,
    callbacks=[early_stop, reduce_lr, per_drug_cb]
)

# --- Evaluate on test set ---
test_results = model.evaluate(x_test, y_test, verbose=0)
test_loss = test_results[0]
test_auc = test_results[2]
print(f"Test loss: {test_loss:.4f}, Test AUC: {test_auc:.4f}")

y_pred_val = model.predict(x_val)

# Use tb-profiler's spec as a floor
tb = pd.read_csv(METRICS / "naiive_perf_metrics.csv",
                 # fix: skip_rows → skiprows
                 skiprows=1).rename(columns=str.lower)

tbsp_raw = [float(str(s).split('/')[0].strip('( '))
            for s in tb['specificity'].tolist()]
tb_spec = {drug: spec for drug, spec in zip(drugs, tbsp_raw)}

tbsen_raw = [float(str(s).split('/')[0].strip('( '))
             for s in tb['sensitivity'].tolist()]
tb_sens = {drug: spec for drug, spec in zip(drugs, tbsen_raw)}

best_thresholds = {}
matched_thresholds = {}
DEFAULT_THR = 0.5

for di, drug in enumerate(drugs):
    mask = ~np.isnan(y_val[:, di])
    yt = y_val[mask, di]
    yp = y_pred_val[mask, di]
    fpr, tpr, thr = roc_curve(y_val[mask, di], y_pred_val[mask, di])
    spec = 1 - fpr
    target_spec = 0.8
    target_sen = tb_sens[drug]

    # check default first
    yb = (yp > DEFAULT_THR).astype(float)
    tn = ((yb == 0) & (yt == 0)).sum()
    fp = ((yb == 1) & (yt == 0)).sum()
    tp = ((yb == 1) & (yt == 1)).sum()
    fn = ((yb == 0) & (yt == 1)).sum()
    spec_d = tn / (tn + fp + 1e-8)
    sens_d = tp / (tp + fn + 1e-8)
    if spec_d >= target_spec and sens_d >= target_sen:
        best_thresholds[drug] = DEFAULT_THR
        continue

    valid = (tpr >= target_sen) & (spec >= target_spec)
    if valid.any():
        # among valid thresholds, pick one maximizing sens
        idx = np.where(valid)[0][np.argmax(tpr[valid])]
        best_thresholds[drug] = float(thr[idx])
        matched_thresholds[drug] = float(thr[idx])
    else:
        # AUC can't even achieve floor — fall back to Youden
        j = tpr - fpr
        best_thresholds[drug] = float(thr[np.argmax(j)])
        matched_thresholds[drug] = 1.0

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
loss_df.to_csv(str(RESULTS / f'metrics/{name}_loss_history.csv'), index=False)
print(f"Saved: {name}_loss_history.csv")

# --- Global learning curves (loss, AUC, F1) ---
epochs = loss_df['epoch']
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 4))

ax1.plot(epochs, loss_df['train_loss'], color='black', label='Train')
ax1.plot(epochs, loss_df['val_loss'],
         color='blue', label='Val', linestyle='--')
ax1.axhline(y=test_loss, color='r', linestyle=':',
            label=f'Test ({test_loss:.4f})')
ax1.set_xlabel('Epoch', fontweight='bold')
ax1.set_ylabel('Loss', fontweight='bold', rotation=0, labelpad=15, va='center')
ax1.set_title('Loss', fontweight='bold')
ax1.legend(frameon=False)

ax2.plot(epochs, loss_df['train_auc'], color='black', label='Train')
ax2.plot(epochs, loss_df['val_auc'], color='blue', label='Val', linestyle='--')
ax2.axhline(y=test_auc, color='r', linestyle=':',
            label=f'Test ({test_auc:.4f})')
ax2.set_xlabel('Epoch', fontweight='bold')
ax2.set_ylabel('AUC', fontweight='bold', rotation=0, labelpad=15, va='center')
ax2.set_title('AUC', fontweight='bold')
ax2.legend(frameon=False)

ax3.plot(epochs, loss_df['train_f1'], color='black', label='Train')
ax3.plot(epochs, loss_df['val_f1'], color='blue', label='Val', linestyle='--')
ax3.axhline(y=test_loss, color='r', linestyle=':',
            label=f'Test ({test_loss:.4f})')
ax3.set_xlabel('Epoch', fontweight='bold')
ax3.set_ylabel('F1', fontweight='bold', rotation=0, labelpad=15, va='center')
ax3.set_title('F1', fontweight='bold')
ax3.legend(frameon=False)

fig.suptitle(f'Global learning curves ({
             tname} SNPs)', fontweight='bold')
fig.tight_layout()
fig.savefig(str(FIGURES / f'{name}_learning_curves.png'), dpi=150)
fig.savefig(str(FIGURES / f'{name}_learning_curves.pdf'), format='pdf')
plt.close(fig)
print(f"Saved: {name}_learning_curves.png")

# --- Per-drug learning curves ---
for drug in drugs:
    drug_epochs = range(1, len(per_drug_cb.history[drug]['loss']) + 1)
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 4))

    ax1.plot(drug_epochs, per_drug_cb.history[drug]['loss'])
    ax1.set_title('Val Loss', fontweight='bold')
    ax1.set_xlabel('Epoch', fontweight='bold')
    ax1.set_ylabel('Loss', fontweight='bold',
                   rotation=0, labelpad=15, va='center')

    ax2.plot(drug_epochs, per_drug_cb.history[drug]['auc'])
    ax2.set_title('Val AUC', fontweight='bold')
    ax2.set_xlabel('Epoch', fontweight='bold')
    ax2.set_ylabel('AUC', fontweight='bold',
                   rotation=0, labelpad=15, va='center')

    ax3.plot(drug_epochs, per_drug_cb.history[drug]['f1'])
    ax3.set_title('Val F1', fontweight='bold')
    ax3.set_xlabel('Epoch', fontweight='bold')
    ax3.set_ylabel('F1', fontweight='bold', rotation=0,
                   labelpad=15, va='center')

    fig.suptitle(
        f'{drug.capitalize()} ({tname} SNPs)', fontweight='bold')
    fig.tight_layout()
    fig.savefig(str(FIGURES / f'{name}_{drug}_learning_curves.png'), dpi=150)
    fig.savefig(
        str(FIGURES / f'{name}_{drug}_learning_curves.pdf'), format='pdf')
    plt.close(fig)
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
per_drug_df = pd.DataFrame(per_drug_rows)
per_drug_df.to_csv(
    str(RESULTS / f'metrics/{name}_per_drug_history.csv'), index=False)
print(f"Saved: {name}_per_drug_history.csv")

# --- Per-drug metrics + confusion matrices ---
y_pred = model.predict(x_test)

pred_df = pd.DataFrame(y_pred, columns=drugs)
pred_df.to_csv(
    str(RESULTS / f'metrics/{name}_trained_predictions.csv'), index=False)

fig, axes = plt.subplots(3, 5, figsize=(18, 10))
axes = axes.flatten()

rows = []
for i, drug in enumerate(drugs):
    mask = ~np.isnan(y_test[:, i])
    yt = y_test[mask, i]
    yp = y_pred[mask, i]
    n = mask.sum()

    ax = axes[i]

    if n == 0 or len(np.unique(yt)) < 2:
        ax.axis('off')
        rows.append({'drug': drug, 'auc': None, 'sensitivity': None,
                     'specificity': None, 'f1': None, 'threshold': None, 'n': n})
        continue

    # --- AUC ---
    auc_m = tf.keras.metrics.AUC()
    auc_m.update_state(yt, yp)
    auc_val = auc_m.result().numpy()

    # --- Threshold sweep ---
    thresh = best_thresholds[drug]
    yb = (yp > thresh).astype(float)
    tp = ((yb == 1) & (yt == 1)).sum()
    fn = ((yb == 0) & (yt == 1)).sum()
    fp = ((yb == 1) & (yt == 0)).sum()
    tn = ((yb == 0) & (yt == 0)).sum()
    sens = tp / (tp + fn + 1e-8)
    spec = tn / (tn + fp + 1e-8)
    prec = tp / (tp + fp + 1e-8)
    f1 = 2 * (prec * sens) / (prec + sens + 1e-8)

    # --- Confusion matrix ---
    cm = np.array([[tn, fp],
                  [fn, tp]])
    ax.imshow(cm, cmap='Blues', vmin=cm.min(), vmax=cm.max())
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Pred S', 'Pred R'], fontweight='bold')
    ax.set_yticklabels(['True S', 'True R'], fontweight='bold')
    ax.set_title(drug.capitalize(), fontsize=9, fontweight='bold')

    # adaptive text colors to maintain readability

    thr = cm.max() / 2.0
    for r in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = 'white' if cm[r, j] > thr else 'black'

            if r == 0 and j == 0:
                dtext = f"{tn}\n(Spec.={round(spec, 3)})"
            elif r == 1 and j == 1:
                dtext = f"{tp}\n(Sens.={round(sens, 3)})"
            else:
                dtext = f"{cm[r, j]}"

            ax.text(j, r, dtext, ha='center',
                    va='center', color=color, fontsize=10, fontweight='bold')

    for pos in ['top', 'bottom', 'left', 'right']:
        ax.spines[pos].set_visible(True)
        ax.spines[pos].set_linewidth(1.2)
        ax.spines[pos].set_color('black')

    rows.append({'drug': drug, 'auc': round(auc_val, 3), 'sensitivity': round(sens, 3),
                 'specificity': round(spec, 3), 'f1': round(f1, 3),
                 'threshold': round(thresh, 2), 'n': n})

metrics_df = pd.DataFrame(rows)


fig.suptitle(
    f"{tname} Confusion matrices (S=Susceptible, R=Resistant)", fontweight='bold', fontsize=12)
fig.tight_layout()
fig.savefig(str(FIGURES / f'{name}_confusion_matrices.png'), dpi=150)
fig.savefig(
    str(FIGURES / f'{name}_confusion_matrices.pdf'), bbox_inches='tight')
plt.close(fig)
print(f"Saved: {name}_confusion_matrices.png")

'''
if name == 'filtered_expert_rules':
    # For finding snps indecies
    with open(RESULTS/'matrices/filtered_expert_rules/filtered_expert_rules_LOOKUP_SNP.json', 'r') as f:
        snp_lu = json.load(f)

    SENS_THR = 0.005
    # Check which snps contribute to an improvement in sensitivity
    final_kernel, final_bias = model.get_layer('large_layer').get_weights()

    good_snps = {d: {} for d in drugs}


    for d in tqdm(drugs, total=len(drugs), desc='sens checking'):
        fsnps = pd.read_csv(
            FILTERED/f'{d}/filtered_{d}_snps.csv')['snp'].tolist()

        di = drugs.index(d)
        # filtered ablated snps
        for snp in fsnps:
            W_ablated = final_kernel.copy()
            if not snp_lu.get(snp):
                continue

            W_ablated[snp_lu[snp], di] = 0.0

            model.get_layer('large_layer').set_weights([W_ablated, final_bias])
            y_pred_test_ablated = model.predict(x_test)

            mask = ~np.isnan(y_test[:, di])
            yt = y_test[mask, di]
            yp = y_pred_test_ablated[mask, di]
            n = mask.sum()

            ax = axes[i]

            # --- Threshold sweep ---
            yb = (yp > best_thresholds[d]).astype(float)
            tp = ((yb == 1) & (yt == 1)).sum()
            fn = ((yb == 0) & (yt == 1)).sum()
            sens = tp / (tp + fn + 1e-8)

            diff = metrics_df.loc[metrics_df['drug']
                                  == d, 'sensitivity'].item() - sens

            if diff >= SENS_THR:
                good_snps[d][snp] = float(diff)

    with open(RESULTS/'essentials/sens_diff_snps.json', 'w') as f:
        json.dump(good_snps, f)
    
    with open(RESULTS/'essentials/sens_diff_snps.json', 'r') as f:
        good_snps = json.load(f)
    # See if metrics improve with only the important snps
    back_select_weight = final_kernel.copy()
    thresholds = best_thresholds.copy()
    for drug, snps in good_snps.items():
        di = drugs.index(drug)
        filter_snps = pd.read_csv(
            FILTERED/f'{drug}/filtered_{drug}_snps.csv')['snp'].tolist()
        for snp in filter_snps:
            if snp not in snp_lu or snp in good_snps.get(drug, {}):
                continue
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

        # --- AUC ---
        auc_m = tf.keras.metrics.AUC()
        auc_m.update_state(yt, yp)
        auc_val = auc_m.result().numpy()

        yb = (yp > thresholds[drug]).astype(float)
        tp = ((yb == 1) & (yt == 1)).sum()
        fn = ((yb == 0) & (yt == 1)).sum()
        fp = ((yb == 1) & (yt == 0)).sum()
        tn = ((yb == 0) & (yt == 0)).sum()
        sens = tp / (tp + fn + 1e-8)
        spec = tn / (tn + fp + 1e-8)
        prec = tp / (tp + fp + 1e-8)
        f1 = 2 * (prec * sens) / (prec + sens + 1e-8)

        r2.append({'drug': drug, 'auc': round(auc_val, 3), 'sensitivity': round(sens, 3),
                   'specificity': round(spec, 3), 'f1': round(f1, 3),
                   'threshold': round(thresholds[drug], 2), 'n': n})
    metrics_df_2 = pd.DataFrame(r2)
    metrics_df_2.to_csv(
    str(RESULTS / f'metrics/{name}_by_drug_metrics.csv'), index=False)
    print(metrics_df_2.to_string(index=False))

    # Restore weights
    model.get_layer('large_layer').set_weights([final_kernel, final_bias])'''

metrics_df.to_csv(
    str(RESULTS / f'metrics/{name}_by_drug_metrics.csv'), index=False)
print(f"Saved: {name}_by_drug_metrics.csv")
print(metrics_df.to_string(index=False))

# --- Save model ---
model.save(str(RESULTS / f'models/trained_models/{name}_trained_model.keras'))
model.export(
    str(RESULTS / f'models/trained_models/{name}_trained_saved_model'))
print(f"Saved: {name}_trained_model.keras")
