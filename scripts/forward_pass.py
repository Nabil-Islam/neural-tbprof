import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Lambda, Concatenate
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
print(drugs)

row_num = len(vcfs)
col_num = len(snp_lu.keys())

dense_shape = [row_num, col_num]
print(dense_shape)

# forward pass
new_snp_vec = Input(shape=(col_num,), sparse=True)

large_dense = Dense(len(drugs), activation='sigmoid', name='large_layer')

large_output = large_dense(new_snp_vec)

pairs = [
    (drugs.index("Isoniazid"), drugs.index("Ethionamide")),
    (drugs.index("Bedaquiline"), drugs.index("Clofazimine")),
    (drugs.index("Levofloxacin"), drugs.index("Moxifloxacin"))
]
paired_indices = set(i for pair in pairs for i in pair)


def slice_cols(idxs):
    return Lambda(lambda x: tf.gather(x, idxs, axis=1))


pair_outputs = {}  # maps index -> combined tensor (shape (batch,1))
for k, (i, j) in enumerate(pairs):
    pair_slice = slice_cols([i, j])(large_output)          # (batch, 2)
    combiner = Dense(1, activation='sigmoid',
                     name=f'pair_combiner_{k}')(pair_slice)  # (batch, 1)
    pair_outputs[i] = combiner
    pair_outputs[j] = combiner   # both slots get the same combined value

# Step 4: reassemble the full 15-length vector, in original order
final_slices = []
for idx in range(len(drugs)):
    if idx in paired_indices:
        final_slices.append(pair_outputs[idx])
    else:
        final_slices.append(slice_cols([idx])(
            large_output))  # (batch,1) untouched

final_output = Concatenate(axis=1, name='final_output')(
    final_slices)  # (batch, 15)

large_model = Model(inputs=new_snp_vec, outputs=final_output)
large_model.summary()

# Update weights
weight_matrix = sp.load_npz(
    results/f'matrices/{name}/{name}_WEIGHTS_AUG.npz').tocoo()

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
df.to_csv(results/f"metrics/cross_{name}_predictions.csv", index=False)

large_model.save(results/f"models/naiive_models/cross_{name}_naiive_model.h5")
