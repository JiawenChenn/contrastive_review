import os
import sys
import time
import numpy as np
import pandas as pd

script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, 'data')
result_dir = os.path.join(script_dir, '..', 'result', 'ccur')
os.makedirs(result_dir, exist_ok=True)

sys.path.insert(0, os.path.join(script_dir, '..', 'utils', 'CCUR'))
from main import *

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

X = target
Y = background
foreground_gene_names = data_pd.columns.values[1:-4]

## CCUR ##
start = time.perf_counter()

cols_contrastive, rows_contrastive, hist = iccur(
    X, Y, cols=10, rows=20, k=10,
    max_iter=100, epsilon=1e-4,
    theta_S=0.90, theta_T=0.90, patience=2,
    verbose=True
)

end = time.perf_counter()
print(f"CCUR runtime: {end - start:.4f} seconds")

# Save selected genes (columns)
selected_genes = foreground_gene_names[cols_contrastive]
print(f"Selected genes: {selected_genes}")

genes_df = pd.DataFrame({
    'gene_index': cols_contrastive,
    'gene_name': selected_genes
})
genes_df.to_csv(os.path.join(result_dir, 'ccur_protein_selected_genes.csv'), index=False)

# Save selected rows
rows_df = pd.DataFrame({
    'row_index': rows_contrastive
})
rows_df.to_csv(os.path.join(result_dir, 'ccur_protein_selected_rows.csv'), index=False)

# Save timing information
with open(os.path.join(result_dir, 'ccur_protein_timing.txt'), 'w') as f:
    f.write(f'CCUR runtime: {end - start:.4f} seconds\n')
