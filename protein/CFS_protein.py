import os
import sys
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pytorch_lightning as pl

script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, 'data')
result_dir = os.path.join(script_dir, '..', 'result', 'cfs')
os.makedirs(result_dir, exist_ok=True)

sys.path.insert(0, os.path.join(script_dir, '..', 'utils', 'CFS'))
from data import LabeledDataset
from CFS_SG import CFS_SG

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

## CFS ##
# Prepare training data: background (label=0) + target (label=1)
labels_train = np.concatenate([np.zeros(background.shape[0]), np.ones(target.shape[0])])
data_train = np.concatenate([background, target])

data_train = torch.from_numpy(data_train).float()
labels_train = torch.from_numpy(labels_train).float()

dataset = LabeledDataset(data_train.numpy(), labels_train.numpy())

input_size = target.shape[1]
output_size = background.shape[1]
batch_size = 128

model = CFS_SG(
    input_size=input_size,
    output_size=output_size,
    hidden=[512, 512],
    k_prime=20,
    lam=0.15,
    lr=1e-3,
    loss_fn=nn.MSELoss()
)

start = time.perf_counter()

loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

trainer = pl.Trainer(max_epochs=10, accelerator='gpu', devices=1)
trainer.fit(model, loader)

end = time.perf_counter()
print(f"CFS runtime: {end - start:.4f} seconds")

# Get selected feature indices
indices = model.get_inds(20)
selected_features = foreground_gene_names[indices]
print(f"Selected features: {selected_features}")

# Save selected features
features_df = pd.DataFrame({
    'feature_index': indices,
    'feature_name': selected_features
})
features_df.to_csv(os.path.join(result_dir, 'cfs_protein_selected_features.csv'), index=False)

# Save timing information
with open(os.path.join(result_dir, 'cfs_protein_timing.txt'), 'w') as f:
    f.write(f'CFS runtime: {end - start:.4f} seconds\n')
