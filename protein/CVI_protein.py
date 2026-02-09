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

data = np.genfromtxt(os.path.join(script_dir, 'data', 'Data_Cortex_Nuclear.csv'), delimiter=',',
                     skip_header=1, usecols=range(1,78), filling_values=0)
classes = np.genfromtxt(os.path.join(script_dir, 'data', 'Data_Cortex_Nuclear.csv'), delimiter=',',
                        skip_header=1, usecols=range(78,81), dtype=None)
import sys

target_idx_A = np.where((classes[:,-1]==b'S/C') & (classes[:,-2]==b'Saline') & (classes[:,-3]==b'Control'))[0]
target_idx_B = np.where((classes[:,-1]==b'S/C') & (classes[:,-2]==b'Saline') & (classes[:,-3]==b'Ts65Dn'))[0]

labels = len(target_idx_A)*[0] + len(target_idx_B)*[1]
target_idx = np.concatenate((target_idx_A,target_idx_B))

target = data[target_idx]

background_idx = np.where((classes[:,-1]==b'C/S') & (classes[:,-2]==b'Saline') & (classes[:,-3]==b'Control'))
background = data[background_idx]

adata = sc.AnnData(X=data,obs=pd.DataFrame(classes))

scvi.external.ContrastiveVI.setup_anndata(adata)

contrastive_vi_model = scvi.external.ContrastiveVI(
    adata, n_salient_latent=2, n_background_latent=2, use_observed_lib_size=False
)

target_idx_all = np.where((classes[:,-1]==b'S/C') & (classes[:,-2]==b'Saline'))

start_time = time.time()
contrastive_vi_model.train(
    background_indices=background_idx[0],
    target_indices=target_idx_all[0],
    early_stopping=True,
    max_epochs=500,
)
end_time = time.time()
elapsed_time = end_time - start_time
print(f'CVI training time: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)')

salient_latent = contrastive_vi_model.get_latent_representation(
    adata, representation_kind="salient"
)

result_dir = os.path.join(script_dir, '..', 'result', 'cvi')
os.makedirs(result_dir, exist_ok=True)

result = pd.DataFrame({'class1': classes[:,-3],
                          'class2': classes[:,-2],
                            'class3': classes[:,-1],
                          'latent_1': salient_latent[:,0],
                          'latent_2': salient_latent[:,1]})

result.to_csv(os.path.join(result_dir, 'cvi_protein_shared2_target2_latent.csv'), index=False)

# Save timing information
with open(os.path.join(result_dir, 'cvi_protein_timing.txt'), 'w') as f:
    f.write(f'CVI training time: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)\n')
