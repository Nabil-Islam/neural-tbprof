"""
Untrained (catalogue-only) single-layer model for one variant set: the
starting weights train_new.py loads, and the untrained predictions
comp_new.py scores (the hollow points in plot.py).

    python scripts/forward_pass_single.py <who|er|filt>

Run after make_index.py and weights.py for the same set. (forward_pass.py
now builds the cross-resistance model instead; this restores the
single-layer version, which had dropped out of the repo.)

Writes
  models/naiive_models/<name>_naiive_model.h5
      kernel (n_variants x 15, drugs in ALPHABETICAL order, as train_new.py's
      labels are) and zero bias, at model_weights/large_layer/large_layer/.
  metrics/<set>/<name>_predictions.csv
      R/S per isolate (rows in vcf_list.txt order), columns in the key order
      of the weights JSON, which is what comp_new.py assumes.

Untrained calls: R if any catalogued variant for that drug is present
(weights 1, bias 0, sigmoid > 0.5), exactly as the previous forward pass.
"""

import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp

SETS = {
    'who':  ('who',                   'who_snps.json',               'who'),
    'er':   ('expert_rules',          'combined_snps.json',          'er'),
    'filt': ('filtered_expert_rules', 'filtered_combined_snps.json', 'filt'),
}

arg = sys.argv[1]
if arg not in SETS:
    raise SystemExit(f"usage: forward_pass_single.py <{'|'.join(SETS)}>")
name, weights_json, subdir = SETS[arg]
results = Path.cwd() / 'results'

with open(results / 'essentials' / weights_json) as f:
    json_drugs = list(json.load(f).keys())          # e.g. 'Amikacin', in file order
alpha_drugs = sorted(json_drugs, key=str.lower)

# weights.py: (drugs in JSON key order) x (variants in LOOKUP order)
W = sp.load_npz(results / f'matrices/{name}/{name}_WEIGHTS_AUG.npz').tocsr()
X = sp.load_npz(results / f'matrices/{name}/{name}_AUG_matrix.npz').tocsr()
if W.shape != (len(json_drugs), X.shape[1]):
    raise SystemExit(f"weights {W.shape} don't match {len(json_drugs)} drugs x "
                     f"{X.shape[1]} matrix columns: rerun make_index.py and weights.py {arg}")

n_runs = sum(1 for line in open(results / 'essentials/vcf_list.txt') if line.strip())
if X.shape[0] != n_runs:
    raise SystemExit(f"matrix has {X.shape[0]} rows but vcf_list.txt has {n_runs}")

if json_drugs != alpha_drugs:
    print("NOTE: weights JSON drug order is not alphabetical:", json_drugs)
    print("      starting weights are reordered to alphabetical for training.")
else:
    print("weights JSON drug order is alphabetical (no reordering needed)")

# --- starting weights for train_new.py: variants x drugs, alphabetical ---
order = [json_drugs.index(d) for d in alpha_drugs]
kernel = W[order].T.toarray().astype(np.float32)          # n_variants x 15
bias = np.zeros(len(alpha_drugs), dtype=np.float32)

out_h5 = results / f'models/naiive_models/{name}_naiive_model.h5'
out_h5.parent.mkdir(parents=True, exist_ok=True)
with h5py.File(out_h5, 'w') as f:
    g = f.create_group('model_weights/large_layer/large_layer')
    g.create_dataset('kernel', data=kernel)
    g.create_dataset('bias', data=bias)
    f.attrs['drugs'] = np.array(alpha_drugs, dtype='S')
    f.attrs['note'] = 'catalogue-initialised single layer; weights only'
print(f"Saved {out_h5}  kernel {kernel.shape}")

# --- untrained predictions for comp_new.py (JSON key order) ---
z = (X @ W.T).toarray()                       # isolates x drugs (JSON order)
calls = np.where(z > 0, 'R', 'S')             # sigmoid(z + 0) > 0.5  <=>  z > 0
out_csv = results / f'metrics/{subdir}/{name}_predictions.csv'
out_csv.parent.mkdir(parents=True, exist_ok=True)
pd.DataFrame(calls, columns=json_drugs).to_csv(out_csv, index=False)
print(f"Saved {out_csv}  {calls.shape}")
print("R calls per drug:", dict(zip(json_drugs, (calls == 'R').sum(0).tolist())))
