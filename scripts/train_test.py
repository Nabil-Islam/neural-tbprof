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

# --- Masked binary crossentropy ---


def masked_binary_crossentropy(y_true, y_pred):
    mask = tf.cast(~tf.math.is_nan(y_true), tf.float32)
    y_true_clean = tf.where(tf.math.is_nan(
        y_true), tf.zeros_like(y_true), y_true)
    loss = tf.keras.backend.binary_crossentropy(y_true_clean, y_pred)
    loss = loss * mask
    return tf.reduce_sum(loss) / (tf.reduce_sum(mask) + 1e-8)


def masked_accuracy(y_true, y_pred):
    mask = tf.cast(~tf.math.is_nan(y_true), tf.float32)
    y_pred_labels = tf.cast(y_pred > 0.5, tf.float32)
    y_true_clean = tf.where(tf.math.is_nan(
        y_true), tf.zeros_like(y_true), y_true)
    correct = tf.cast(tf.equal(y_true_clean, y_pred_labels), tf.float32)
    return tf.reduce_sum(correct * mask) / (tf.reduce_sum(mask) + 1e-8)


# --- Compile ---
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss=masked_binary_crossentropy,
    metrics=[masked_accuracy]
)

# --- Train ---
early_stop = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)
reduce_lr = keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,
    patience=3,
    min_lr=1e-5)

test_loss = model.evaluate(x_test, y_test, verbose=0)[0]

history = model.fit(
    x_train, y_train,
    epochs=25,
    validation_split=0.2,
    batch_size=256,
    callbacks=[early_stop, reduce_lr]
)

print("History keys:", history.history.keys())
print("Epochs run:", len(history.history['loss']))
print("Train loss:", history.history['loss'])
print("Val loss:", history.history['val_loss'])

# --- Save loss history ---
loss_df = pd.DataFrame({
    'epoch':      range(1, len(history.history['loss']) + 1),
    'train_loss': history.history['loss'],
    'val_loss':   history.history['val_loss']
})
loss_df['test_loss'] = test_loss
loss_df.to_csv(str(RESULTS / f'metrics/{name}_loss_history.csv'), index=False)
print(f"Saved: {name}_loss_history.csv")

# --- Learning curves ---
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(loss_df['epoch'], loss_df['train_loss'], label='Train loss')
ax.plot(loss_df['epoch'], loss_df['val_loss'],
        label='Val loss', linestyle='--')
ax.axhline(y=test_loss, color='r', linestyle=':',
           label=f'Test loss({test_loss:.4f})')
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
ax.set_title(f'{name} Learning curves')
ax.legend()
fig.tight_layout()
fig.savefig(str(FIGURES / f'{name}_learning_curves.png'), dpi=150)
plt.close(fig)
print(f"Saved: {name}_learning_curves.png")

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

    im = ax.imshow(cm, cmap='Blues')
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Pred S', 'Pred R'])
    ax.set_yticklabels(['True S', 'True R'])
    ax.set_title(drug, fontsize=9)
    for r in range(2):
        for c in range(2):
            ax.text(c, r, str(cm[r, c]), ha='center', va='center', fontsize=10)

    rows.append({'drug': drug, 'auc': round(auc_val, 3), 'sensitivity': round(best_sens, 3),
                 'specificity': round(best_spec, 3), 'f1': round(best_f1, 3),
                 'threshold': round(best_thresh, 2), 'n': n})

fig.suptitle('Confusion matrices (S=susceptible, R=resistant)')
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
