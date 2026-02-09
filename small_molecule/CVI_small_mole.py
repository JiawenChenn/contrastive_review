import os
import tempfile
import time

import numpy as np
import requests
import scanpy as sc
import scvi
import seaborn as sns
import torch
import pandas as pd
scvi.settings.seed = 0
print("Last run with scvi-tools version:", scvi.__version__)

sc.set_figure_params(figsize=(6, 6), frameon=False)
sns.set_theme()
torch.set_float32_matmul_precision("high")

script_dir = os.path.dirname(os.path.abspath(__file__))

# Load small molecule data
data_dir = os.path.join(script_dir, 'data')

background = pd.read_csv(os.path.join(data_dir, 'count', 'background_count.csv'), index_col=0).values
foreground = pd.read_csv(os.path.join(data_dir, 'count', 'foreground_count.csv'), index_col=0).values
labels_df = pd.read_csv(os.path.join(data_dir, 'labels.csv'))
foreground_label = labels_df['label'].values

data = np.concatenate([background, foreground], axis=0)
background_idx = np.arange(0, background.shape[0])
target_idx = np.arange(background.shape[0], data.shape[0])

adata = sc.AnnData(X=data)

print('shape of target data:', foreground.shape)
print('shape of background data:', background.shape)

scvi.external.ContrastiveVI.setup_anndata(adata)

contrastive_vi_model = scvi.external.ContrastiveVI(
    adata, n_salient_latent=2, n_background_latent=2, use_observed_lib_size=False
)

start_time = time.time()
contrastive_vi_model.train(
    background_indices=background_idx,
    target_indices=target_idx,
    early_stopping=True,
    max_epochs=500,
)
end_time = time.time()
elapsed_time = end_time - start_time
print(f'CVI training time: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)')

target_adata = adata[target_idx].copy()

salient_latent = contrastive_vi_model.get_latent_representation(
    target_adata, representation_kind="salient"
)

result_dir = os.path.join(script_dir, '..', 'result', 'cvi')
os.makedirs(result_dir, exist_ok=True)

result = pd.DataFrame({'label': foreground_label,
                          'latent_1': salient_latent[:,0],
                          'latent_2': salient_latent[:,1]})

result.to_csv(os.path.join(result_dir, 'cvi_small_mole_shared2_target2_latent.csv'), index=False)

# Save timing information
with open(os.path.join(result_dir, 'cvi_small_mole_timing.txt'), 'w') as f:
    f.write(f'CVI training time: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)\n')
