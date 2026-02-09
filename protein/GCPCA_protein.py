import os
import sys
import time
import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score
import generalized_contrastive_PCA as gcPCA

script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, 'data')
result_dir = os.path.join(script_dir, '..', 'result', 'gcpca')
os.makedirs(result_dir, exist_ok=True)

## Read in data ##
data = np.genfromtxt(os.path.join(data_dir, 'Data_Cortex_Nuclear.csv'), delimiter=',',
                     skip_header=1, usecols=range(1,78), filling_values=0)
classes = np.genfromtxt(os.path.join(data_dir, 'Data_Cortex_Nuclear.csv'), delimiter=',',
                        skip_header=1, usecols=range(78,81), dtype=None)

data_pd = pd.read_csv(os.path.join(data_dir, "Data_Cortex_Nuclear.csv"))

# Handle both string and bytes comparisons from genfromtxt dtype=None
sample_val = classes[0, -1]
if isinstance(sample_val, bytes):
    SC, CS, Saline, Control, Ts65Dn = b'S/C', b'C/S', b'Saline', b'Control', b'Ts65Dn'
else:
    SC, CS, Saline, Control, Ts65Dn = 'S/C', 'C/S', 'Saline', 'Control', 'Ts65Dn'

target_idx_A = np.where((classes[:,-1]==SC) & (classes[:,-2]==Saline) & (classes[:,-3]==Control))[0]
target_idx_B = np.where((classes[:,-1]==SC) & (classes[:,-2]==Saline) & (classes[:,-3]==Ts65Dn))[0]

labels = len(target_idx_A)*[0] + len(target_idx_B)*[1]
target_idx = np.concatenate((target_idx_A, target_idx_B))

target = data[target_idx]

background_idx = np.where((classes[:,-1]==CS) & (classes[:,-2]==Saline) & (classes[:,-3]==Control))
background = data[background_idx]

foreground_gene_names = data_pd.columns.values[1:-4]

## GCPCA ##
Ra = target.copy()
Rb = background.copy()

start = time.perf_counter()

model = gcPCA.gcPCA(method='v4.1', normalize_flag=True)
model.fit(Ra, Rb)

end = time.perf_counter()
print(f"GCPCA runtime: {end - start:.4f} seconds")

# Scores (first 2 components)
Z_gc = model.Ra_scores_[:, :2]

# Save results
result = pd.DataFrame({
    'class1': classes[target_idx, -3],
    'class2': classes[target_idx, -2],
    'class3': classes[target_idx, -1],
    'latent_1': Z_gc[:, 0],
    'latent_2': Z_gc[:, 1]
})
result.to_csv(os.path.join(result_dir, 'gcpca_protein_latent.csv'), index=False)

# Save timing information
with open(os.path.join(result_dir, 'gcpca_protein_timing.txt'), 'w') as f:
    f.write(f'GCPCA runtime: {end - start:.4f} seconds\n')

# Silhouette score
sil_score = silhouette_score(Z_gc, labels, metric='euclidean')
print(f"Silhouette Score: {sil_score:.4f}")

with open(os.path.join(result_dir, 'gcpca_protein_silhouette.txt'), 'w') as f:
    f.write(f'Silhouette Score: {sil_score:.4f}\n')
