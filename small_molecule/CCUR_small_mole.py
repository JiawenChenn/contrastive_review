import sys
import os
import numpy as np
import pandas as pd
import time

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(script_dir, '..', 'utils', 'CCUR'))

from main import *

# Load small molecule data
data_dir = os.path.join(script_dir, 'data')

gene_names = pd.read_csv(os.path.join(data_dir, "genes.csv"))["gene"].to_numpy()
cell_names = pd.read_csv(os.path.join(data_dir, "foreground_cells.csv"))["cell"].to_numpy()
foreground = pd.read_csv(os.path.join(data_dir, "foreground_matrix.csv"), index_col=0)
background = pd.read_csv(os.path.join(data_dir, "background_matrix.csv"), index_col=0)

cell_types = pd.read_csv(os.path.join(data_dir, "cell_types_foreground.csv"))['cell_type'].values

X = foreground.to_numpy()
Y = background.to_numpy()

print('shape of foreground data:', X.shape)
print('shape of background data:', Y.shape)

# Run iCCUR
start = time.perf_counter()

cols_contrastive, rows_contrastive, hist = iccur(
    X, Y, cols=20, rows=200, k=7,
    max_iter=100, epsilon=1e-6,
    theta_S=0.90, theta_T=0.90, patience=2,
    verbose=True
)

end = time.perf_counter()
print(f"CCUR runtime: {end - start:.4f} seconds")

# Print selected genes
selected_genes = gene_names[cols_contrastive]
print("Selected contrastive genes:")
for g in selected_genes:
    print(f"  {g}")

# Save results
result_dir = os.path.join(script_dir, '..', 'result', 'ccur')
os.makedirs(result_dir, exist_ok=True)

# Save selected genes (columns)
cols_df = pd.DataFrame({
    'gene_index': cols_contrastive,
    'gene_name': gene_names[cols_contrastive]
})
cols_df.to_csv(os.path.join(result_dir, 'ccur_small_mole_selected_genes.csv'), index=False)

# Save selected rows
rows_df = pd.DataFrame({
    'row_index': rows_contrastive
})
rows_df.to_csv(os.path.join(result_dir, 'ccur_small_mole_selected_rows.csv'), index=False)

# Save convergence history
hist_df = pd.DataFrame({'history': hist})
hist_df.to_csv(os.path.join(result_dir, 'ccur_small_mole_history.csv'), index=False)

# Save timing
elapsed_time = end - start
with open(os.path.join(result_dir, 'ccur_small_mole_timing.txt'), 'w') as f:
    f.write(f'CCUR runtime: {elapsed_time:.4f} seconds\n')

print('Results saved.')
