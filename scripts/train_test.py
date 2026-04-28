import tensorflow as tf
import tf_keras as keras
import numpy as np
import h5py
import pandas as pd
import scipy.sparse as sp
from pathlib import Path

# --- Paths ---
SCRIPTS = Path.cwd()
RESULTS = SCRIPTS / 'results'

drugs = ['amikacin', 'bedaquiline', 'capreomycin', 'clofazimine', 'delamanid',
         'ethambutol', 'ethionamide', 'isoniazid', 'kanamycin', 'levofloxacin',
         'linezolid', 'moxifloxacin', 'pyrazinamide', 'rifampicin', 'streptomycin']

# --- Load feature matrix ---
matrix = sp.load_npz(RESULTS / 'expert_rules_AUG_matrix.npz')
X = matrix.toarray().astype(np.float32)

# --- Load label matrix ---
label_matrix = pd.read_csv(RESULTS / 'labelled_vcfs.csv')
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
with h5py.File(str(RESULTS / 'forward_model.h5'), 'r') as f:
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

history = model.fit(
    x_train, y_train,
    epochs=50,
    validation_split=0.2,
    batch_size=264,
    callbacks=[early_stop, reduce_lr]
)

# --- Save loss history ---
loss_df = pd.DataFrame({
    'epoch':      range(1, len(history.history['loss']) + 1),
    'train_loss': history.history['loss'],
    'val_loss':   history.history['val_loss']
})
loss_df.to_csv(str(RESULTS / 'loss_history.csv'), index=False)
print("Saved: loss_history.csv")

# --- Per-drug metrics ---
y_pred = model.predict(x_test)
pred_df = pd.DataFrame(y_pred, columns=drugs)
pred_df.to_csv(str(RESULTS / 'trained_predictions.csv'), index=False)


rows = []
for i, drug in enumerate(drugs):
    mask = ~np.isnan(y_test[:, i])
    yt = y_test[mask, i]
    yp = y_pred[mask, i]
    n = mask.sum()

    if n == 0 or len(np.unique(yt)) < 2:
        rows.append({'drug': drug, 'auc': None, 'sensitivity': None,
                     'specificity': None, 'f1': None, 'threshold': None, 'n': n})
        continue

    auc_m = tf.keras.metrics.AUC()
    auc_m.update_state(yt, yp)
    auc_val = auc_m.result().numpy()

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

    rows.append({'drug': drug, 'auc': round(auc_val, 3), 'sensitivity': round(best_sens, 3),
                 'specificity': round(best_spec, 3), 'f1': round(best_f1, 3),
                 'threshold': round(best_thresh, 2), 'n': n})

metrics_df = pd.DataFrame(rows)
metrics_df.to_csv(str(RESULTS / 'drug_metrics.csv'), index=False)
print("Saved: drug_metrics.csv")
print(metrics_df.to_string(index=False))

# --- Save model ---
model.save(str(RESULTS / 'trained_model.keras'))
print("Saved: trained_model.keras")
