import os
import sys
import time
import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score
from pcpca import PCPCA

script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, 'data')
result_dir = os.path.join(script_dir, '..', 'result', 'pcpca')
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

## PCPCA ##
# Center and scale
foreground_centered = (target - np.mean(target, axis=0)) / np.std(target, axis=0)
background_centered = (background - np.mean(background, axis=0)) / np.std(background, axis=0)

start = time.perf_counter()

pcpca = PCPCA(gamma=0.6, n_components=2)
pcpca.fit(foreground_centered.T, background_centered.T)

end = time.perf_counter()
print(f"PCPCA runtime: {end - start:.4f} seconds")

X_reduced, Y_reduced = pcpca.fit_transform(foreground_centered.T, background_centered.T)
X_reduced = np.array(X_reduced)

# Save results
result = pd.DataFrame({
    'class1': classes[target_idx, -3],
    'class2': classes[target_idx, -2],
    'class3': classes[target_idx, -1],
    'latent_1': X_reduced.T[:, 0],
    'latent_2': X_reduced.T[:, 1]
})
result.to_csv(os.path.join(result_dir, 'pcpca_protein_latent.csv'), index=False)

# Save timing information
with open(os.path.join(result_dir, 'pcpca_protein_timing.txt'), 'w') as f:
    f.write(f'PCPCA runtime: {end - start:.4f} seconds\n')

# Silhouette score
sil_score = silhouette_score(X_reduced.T, labels, metric='euclidean')
print(f"Silhouette Score: {sil_score:.4f}")

with open(os.path.join(result_dir, 'pcpca_protein_silhouette.txt'), 'w') as f:
    f.write(f'Silhouette Score: {sil_score:.4f}\n')
