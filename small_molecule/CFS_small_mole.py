import sys
import os
import numpy as np
import pandas as pd
import time
import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(script_dir, '..', 'utils', 'CFS'))

from CFS_SG import CFS_SG

# Load small molecule data
data_dir = os.path.join(script_dir, 'data')

foreground = pd.read_csv(os.path.join(data_dir, 'foreground_matrix.csv'))
background = pd.read_csv(os.path.join(data_dir, 'background_matrix.csv'))

foreground = foreground.set_index('Unnamed: 0')
background = background.set_index('Unnamed: 0')

gene_names = foreground.columns.to_numpy()
cell_names = foreground.index.to_numpy()

print('shape of foreground data:', foreground.shape)
print('shape of background data:', background.shape)

# Prepare training data: background label=0, foreground label=1
labels_train = np.concatenate([np.zeros(background.shape[0]), np.ones(foreground.shape[0])])
data_train = np.concatenate([background, foreground])

data_train_tensor = torch.tensor(data_train, dtype=torch.float32)
labels_train_tensor = torch.tensor(labels_train, dtype=torch.long)

dataset = TensorDataset(data_train_tensor, labels_train_tensor)

input_size = foreground.shape[1]
output_size = background.shape[1]
batch_size = 128

# Build CFS model
model = CFS_SG(
    input_size=input_size,
    output_size=output_size,
    hidden=[512, 512],
    k_prime=20,
    lam=0.175,
    lr=1e-3,
    loss_fn=nn.MSELoss()
)

# Train
start = time.perf_counter()

loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

trainer = pl.Trainer(
    accelerator="gpu",
    devices=1,
    max_epochs=10
)
trainer.fit(model, loader)

end = time.perf_counter()

print(f"CFS runtime: {end - start:.4f} seconds")

# Get selected feature indices
indices = model.get_inds(10)
selected_genes = gene_names[indices]
print("Selected genes:")
for g in selected_genes:
    print(f"  {g}")

# Save results
result_dir = os.path.join(script_dir, '..', 'result', 'cfs')
os.makedirs(result_dir, exist_ok=True)

# Save selected features
features_df = pd.DataFrame({
    'gene_index': indices,
    'gene_name': selected_genes
})
features_df.to_csv(os.path.join(result_dir, 'cfs_small_mole_selected_features.csv'), index=False)

# Save timing
elapsed_time = end - start
with open(os.path.join(result_dir, 'cfs_small_mole_timing.txt'), 'w') as f:
    f.write(f'CFS runtime: {elapsed_time:.4f} seconds\n')

print('Results saved.')
