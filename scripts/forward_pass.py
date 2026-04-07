import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
import numpy as np
from pathlib import Path
import json
import scipy.sparse as sp
import pandas as pd
import sys
import tensorflowjs as tfjs

workflow = Path.cwd()
results = workflow/"results"
data = workflow/"data"
outfile = results/"NEW_fordPass.json"
arg = sys.argv[1]

vcfs = []

with open(results/"vcf_list.txt") as f:
    for vcf in f:
        vcfs.append(vcf.strip('\n'))

if arg == "er":
    with open(results/'combined_snps.json') as f:
        snp_dict = json.load(f)
    name = "expert_rules"

if arg == "who":
    with open(results/'who_snps.json', 'r') as f:
        snp_dict = json.load(f)
    name = "who"

with open(results/f"{name}_LOOKUP_SNP.json") as f:
    snp_lu = json.load(f)


drugs = list(snp_dict.keys())

row_num = len(vcfs)
col_num = len(snp_lu.keys())

dense_shape = [row_num, col_num]
print(dense_shape)

# forward pass
new_snp_vec = Input(shape=(col_num,), sparse=True)

large_dense = Dense(len(drugs), activation='sigmoid', name='large_layer')

large_output = large_dense(new_snp_vec)

large_model = Model(inputs=new_snp_vec, outputs=large_output)

print(large_model.summary())

# Update weights
weight_matrix = sp.load_npz(results/f'{name}_WEIGHTS_AUG.npz').tocoo()

indices = np.vstack((weight_matrix.row, weight_matrix.col)).T
values = weight_matrix.data
shape = weight_matrix.shape

tf_weights = tf.sparse.SparseTensor(
    indices=indices.astype(np.int64),
    values=values.astype(np.float32),
    dense_shape=np.array(shape, dtype=np.int64)
)

tf_weights = tf.sparse.reorder(tf_weights)

dense_kernel = tf.sparse.to_dense(tf_weights)

dense_kernel = tf.transpose(dense_kernel)

bias = np.zeros(len(drugs), dtype=np.float32)

large_dense.set_weights([dense_kernel.numpy(), bias])

# Make new layer
matrix = sp.load_npz(results/f"{name}_AUG_matrix.npz").tocoo()

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

if arg == "er":
    df.to_csv(results/"er_predictions.csv", index=False)

if arg == "who":
    df.to_csv(results/"who_predictions.csv", index=False)

dense_input = Input(shape=(col_num,), sparse=False)
dense_output = large_dense(dense_input)  # reuse trained layer, weights intact
dense_model = Model(inputs=dense_input, outputs=dense_output)

# save in all formats
dense_model.save(results / 'forward_model.h5')
dense_model.save(str(results / 'forward_model.keras'))
tf.saved_model.save(dense_model, str(results / 'forward_model_saved'))


converter = tf.lite.TFLiteConverter.from_keras_model(large_model)
tflite_model = converter.convert()

with open(results / 'forward_model.tflite', 'wb') as f:
    f.write(tflite_model)

print(f"Sved forward_model.tflite")

tfjs.converters.save_keras_model(large_model, results / "tfjs_model")
print(f"Saved TFJS model to {results / 'tfjs_model'}")
