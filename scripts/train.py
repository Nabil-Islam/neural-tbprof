import sys
from pathlib import Path
import matplotlib.pyplot as plt
import tensorflow as tf
import tf_keras as keras
import numpy as np
import h5py
import pandas as pd
import scipy.sparse as sp
import matplotlib
matplotlib.use('Agg')  # headless backend for HPC

arg = sys.argv[1]

if arg == 'who':
    name = 'who'
if arg == 'er':
    name = 'expert_rules'
if arg == 'filt':
    name = 'filtered_expert_rules'

# --- Paths ---
SCRIPTS = Path.cwd()
RESULTS = SCRIPTS / 'results'
FIGURES = RESULTS / 'figures'
FIGURES.mkdir(parents=True, exist_ok=True)

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
    kernel = f['model_weights/large_layer/large_layer/kernel'][:]
    bias = f['model_weights/large_layer/large_layer/bias'][:]

# --- Build model ---
inputs = keras.Input(shape=(X.shape[1],))
outputs = keras.layers.Dense(
    15, activation='sigmoid', name='large_layer')(inputs)
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
    optimizer=keras.optimizers.Adam(learning_rate=1e-5),
    loss=masked_binary_crossentropy,
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
    patience=20,
    restore_best_weights=True
)
reduce_lr = keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,
    patience=4,
    min_lr=1e-6
)

# --- Train ---
history = model.fit(
    x_tr, y_tr,
    epochs=300,
    validation_data=(x_val, y_val),
    batch_size=64,
    callbacks=[early_stop, reduce_lr, per_drug_cb]
)

# --- Evaluate on test set ---
test_results = model.evaluate(x_test, y_test, verbose=0)
test_loss = test_results[0]
test_auc = test_results[2]
print(f"Test loss: {test_loss:.4f}, Test AUC: {test_auc:.4f}")

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

ax1.plot(epochs, loss_df['train_loss'], label='Train')
ax1.plot(epochs, loss_df['val_loss'], label='Val', linestyle='--')
ax1.axhline(y=test_loss, color='r', linestyle=':',
            label=f'Test ({test_loss:.4f})')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.set_title('Loss')
ax1.legend()

ax2.plot(epochs, loss_df['train_auc'], label='Train')
ax2.plot(epochs, loss_df['val_auc'], label='Val', linestyle='--')
ax2.axhline(y=test_auc, color='r', linestyle=':',
            label=f'Test ({test_auc:.4f})')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('AUC')
ax2.set_title('AUC')
ax2.legend()

ax3.plot(epochs, loss_df['train_f1'], label='Train')
ax3.plot(epochs, loss_df['val_f1'], label='Val', linestyle='--')
ax3.set_xlabel('Epoch')
ax3.set_ylabel('F1')
ax3.set_title('F1')
ax3.legend()

fig.suptitle(f'Global learning curves — {name.replace("_", " ").capitalize()}')
fig.tight_layout()
fig.savefig(str(FIGURES / f'{name}_learning_curves.png'), dpi=150)
plt.close(fig)
print(f"Saved: {name}_learning_curves.png")

# --- Per-drug learning curves ---
for drug in drugs:
    drug_epochs = range(1, len(per_drug_cb.history[drug]['loss']) + 1)
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 4))

    ax1.plot(drug_epochs, per_drug_cb.history[drug]['loss'])
    ax1.set_title('Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')

    ax2.plot(drug_epochs, per_drug_cb.history[drug]['auc'])
    ax2.set_title('Val AUC')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('AUC')

    ax3.plot(drug_epochs, per_drug_cb.history[drug]['f1'])
    ax3.set_title('Val F1')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('F1')

    fig.suptitle(f'{drug.capitalize()} — {
                 name.replace("_", " ").capitalize()}')
    fig.tight_layout()
    fig.savefig(str(FIGURES / f'{name}_{drug}_learning_curves.png'), dpi=150)
    plt.close(fig)
    print(f"Saved: {name}_{drug}_learning_curves.png")

# --- Save per-drug history to CSV ---
per_drug_rows = []
for drug in drugs:
    for ep, (l, a, f) in enumerate(zip(
            per_drug_cb.history[drug]['loss'],
            per_drug_cb.history[drug]['auc'],
            per_drug_cb.history[drug]['f1']), start=1):
        per_drug_rows.append({'drug': drug, 'epoch': ep, 'val_loss': l,
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


def confusion_matrix_np(yt, yb):
    """Pure numpy 2x2 confusion matrix [[TN, FP], [FN, TP]]"""
    tp = int(((yb == 1) & (yt == 1)).sum())
    fn = int(((yb == 0) & (yt == 1)).sum())
    fp = int(((yb == 1) & (yt == 0)).sum())
    tn = int(((yb == 0) & (yt == 0)).sum())
    return np.array([[tn, fp], [fn, tp]])


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
    best_thresh, best_f1, best_sens, best_spec = 0.5, 0.0, 0.0, 0.0
    for thresh in np.arange(0.05, 0.95, 0.05):
        yb = (yp > thresh).astype(float)
        tp = ((yb == 1) & (yt == 1)).sum()
        fn = ((yb == 0) & (yt == 1)).sum()
        fp = ((yb == 1) & (yt == 0)).sum()
        tn = ((yb == 0) & (yt == 0)).sum()
        sens = tp / (tp + fn + 1e-8)
        spec = tn / (tn + fp + 1e-8)
        prec = tp / (tp + fp + 1e-8)
        f1 = 2 * (prec * sens) / (prec + sens + 1e-8)
        if f1 > best_f1:
            best_f1, best_thresh, best_sens, best_spec = f1, thresh, sens, spec

    # --- Confusion matrix ---
    yb_best = (yp > best_thresh).astype(float)
    cm = confusion_matrix_np(yt, yb_best)

    ax.imshow(cm, cmap='Blues')
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Pred S', 'Pred R'])
    ax.set_yticklabels(['True S', 'True R'])
    ax.set_title(drug, fontsize=9)

    # adaptive text colors to maintain readability
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = 'white' if cm[i, j] > thresh else 'black'
            ax.text(j, i, str(cm[i, j]), ha='center',
                    va='center', color=color, fontsize=8)

    rows.append({'drug': drug, 'auc': round(auc_val, 3), 'sensitivity': round(best_sens, 3),
                 'specificity': round(best_spec, 3), 'f1': round(best_f1, 3),
                 'threshold': round(best_thresh, 2), 'n': n})

fig.suptitle(
    f"{name.replace('_', ' ').capitalize()} Confusion matrices (S=susceptible, R=resistant)")
fig.tight_layout()
fig.savefig(str(FIGURES / f'{name}_confusion_matrices.png'), dpi=150)
plt.close(fig)
print(f"Saved: {name}_confusion_matrices.png")

metrics_df = pd.DataFrame(rows)
metrics_df.to_csv(
    str(RESULTS / f'metrics/{name}_by_drug_metrics.csv'), index=False)
print(f"Saved: {name}_by_drug_metrics.csv")
print(metrics_df.to_string(index=False))

# --- Save model ---
model.save(str(RESULTS / f'models/trained_models/{name}_trained_model.keras'))
model.export(
    str(RESULTS / f'models/trained_models/{name}_trained_saved_model'))
print(f"Saved: {name}_trained_model.keras")
