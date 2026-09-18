import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
import numpy as np
from pathlib import Path
import json
import scipy.sparse as sp
import pandas as pd
import sys

workflow = Path.cwd()
results = workflow/"results"
data = workflow/"data"
outfile = results/"NEW_fordPass.json"
arg = sys.argv[1]

vcfs = []

with open(results/"essentials/vcf_list.txt") as f:
    for vcf in f:
        vcfs.append(vcf.strip('\n'))

if arg == "er":
    with open(results/'essentials/combined_snps.json') as f:
        snp_dict = json.load(f)
    name = "expert_rules"

if arg == "filt":
    with open(results/'essentials/filtered_combined_snps.json') as f:
        snp_dict = json.load(f)
    name = "filtered_expert_rules"

if arg == "who":
    with open(results/'essentials/who_snps.json', 'r') as f:
        snp_dict = json.load(f)
    name = "who"

with open(results/f"matrices/{name}/{name}_LOOKUP_SNP.json") as f:
    snp_lu = json.load(f)

drugs = list(snp_dict.keys())

row_num = len(vcfs)
col_num = len(snp_lu.keys())

dense_shape = [row_num, col_num]
cr_dic = {
    "levofloxacin": "moxifloxacin",
    "moxifloxacin": "levofloxacin",
    "bedaquiline": "clofazimine",
    "clofazimine": "bedaquiline",
    "isoniazid": "ethionamide",
    "ethionamide": "isoniazid"
}
print(dense_shape)

# forward pass
snp_vec = Input(shape=(col_num,), sparse=True)

# Make layers for cross resistance
cr_layer = Dense(len(cr_dic), activation='sigmoid', name='cr_layer')
cr_info = cr_layer(snp_vec)

# Cross over different drugs LFX<->MFX, BDQ<->CFZ, INH<->ETO
p_idx = [1, 0, 3, 2, 5, 4]
cr_reorder = tf.keras.layers.Lambda(
    lambda t: tf.gather(t, p_idx, axis=1)
)(cr_info)

cr_pred = Dense(len(cr_dic), activation='sigmoid', name='cr_pred')
comb_cr = tf.keras.layers.Concatenate(axis=1)([snp_vec, cr_reorder])
cr_out = cr_pred(comb_cr)

# Prediction for non-cr drugs
r_len = len(drugs) - len(cr_dic)
r_layer = Dense(r_len, activation='sigmoid', name='r_layer')
r_out = r_layer(snp_vec)

large_output = tf.keras.layers.Concatenate(axis=1)([r_out, cr_out])

large_model = Model(inputs=snp_vec, outputs=large_output)

print(large_model.summary())

# Load weights for cross-R
cr_w_m = sp.load_npz(
    results/f'matrices/{name}/{name}_CR_WEIGHTS_AUG.npz').tocoo()
cr_w_m_one = sp.load_npz(
    results/f'matrices/{name}/{name}_CR+1_WEIGHTS_AUG.npz').tocoo()

idx = np.vstack((cr_w_m.row, cr_w_m.col)).T
val = cr_w_m.data
shp = cr_w_m.shape

cr_weights = tf.sparse.SparseTensor(
    indices=idx.astype(np.int64),
    values=val.astype(np.float32),
    dense_shape=np.array(shp, dtype=np.int64)
)

idx = np.vstack((cr_w_m_one.row, cr_w_m_one.col)).T
val = cr_w_m_one.data
shp = cr_w_m_one.shape


cr_one_weights = tf.sparse.SparseTensor(
    indices=idx.astype(np.int64),
    values=val.astype(np.float32),
    dense_shape=np.array(shp, dtype=np.int64)
)

cr_weights = tf.sparse.reorder(cr_weights)
dense_kernel = tf.sparse.to_dense(cr_weights)
dense_kernel = tf.transpose(dense_kernel)
bias = np.zeros(len(cr_dic), dtype=np.float32)
cr_layer.set_weights([dense_kernel.numpy(), bias])

cr_one_weights = tf.sparse.reorder(cr_one_weights)
dense_kernel = tf.sparse.to_dense(cr_one_weights)
dense_kernel = tf.transpose(dense_kernel)
cr_pred.set_weights([dense_kernel.numpy(), bias])

# Load weights for non-cross resistance
w_m = sp.load_npz(
    results/f'matrices/{name}/{name}_WEIGHTS_AUG.npz').tocoo()

idx = np.vstack((w_m.row, w_m.col)).T
val = w_m.data
shp = w_m.shape

r_weights = tf.sparse.SparseTensor(
    indices=idx.astype(np.int64),
    values=val.astype(np.float32),
    dense_shape=np.array(shp, dtype=np.int64)
)

r_weights = tf.sparse.reorder(r_weights)
dense_kernel = tf.sparse.to_dense(r_weights)
dense_kernel = tf.transpose(dense_kernel)
bias = np.zeros(len(drugs)-len(cr_dic), dtype=np.float32)
r_layer.set_weights([dense_kernel.numpy(), bias])

# Import input tensor created from VCF files
matrix = sp.load_npz(results/f"matrices/{name}/{name}_AUG_matrix.npz").tocoo()

indices = np.vstack((matrix.row, matrix.col)).T.astype(np.int64)
values = matrix.data.astype(np.float32)
dense_shape = np.array(matrix.shape, dtype=np.int64)


snp_sparse = tf.sparse.SparseTensor(
    indices=indices,
    values=values,
    dense_shape=dense_shape
)

# Reorder for safe operations
snp_sparse = tf.sparse.reorder(snp_sparse)

prediction = large_model(snp_sparse)

prediction = prediction.numpy().tolist()

for vcf in prediction:
    for i, pred in enumerate(vcf):
        vcf[i] = 'R' if pred > 0.5 else 'S'

df = pd.DataFrame(prediction)
df.to_csv(results/f"metrics/{name}_CR_predictions.csv", index=False)

large_model.save(results/f"models/naiive_models/{name}_CR_naiive_model.h5")
