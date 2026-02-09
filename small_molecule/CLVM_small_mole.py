#conda activate tfp_env
import matplotlib
matplotlib.use('TkAgg')

import numpy as np
import os
import sys
import time

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(script_dir, '..', 'utils', 'clvm', 'contrastive-LVM'))

from clvm_tfp import clvm

from utils.factor_plot import factor_plot

import pandas as pd

# Load small molecule data
data_dir = os.path.join(script_dir, 'data')

background = pd.read_csv(os.path.join(data_dir, 'background_matrix.csv'), index_col=0).values
foreground = pd.read_csv(os.path.join(data_dir, 'foreground_matrix.csv'), index_col=0).values
labels_df = pd.read_csv(os.path.join(data_dir, 'labels.csv'))
foreground_label = labels_df['label'].values

target = foreground

print('shape of target data:', target.shape)
print('shape of background data:', background.shape)

result_dir = os.path.join(script_dir, '..', 'result', 'clvm')
os.makedirs(result_dir, exist_ok=True)

start_time = time.time()
model_0 = clvm(target, background, k_shared=2, k_target=2)
t_clvm = model_0.variational_inference(num_epochs=10000, plot=False, fp=os.path.join(result_dir, 'small_mole_shared2_target2_'))
end_time = time.time()
elapsed_time = end_time - start_time
print(f'CLVM training time: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)')

np.savetxt(os.path.join(result_dir, 'clvm_small_mole_shared2_target2_loading_w.txt'), model_0.w_hat)

result = pd.DataFrame({'classes': foreground_label,
                       'latent_1': t_clvm[:,0],
                       'latent_2': t_clvm[:,1]})

result.to_csv(os.path.join(result_dir, 'clvm_small_mole_shared2_target2_latent.csv'), index=False)

# Save timing information
with open(os.path.join(result_dir, 'clvm_small_mole_timing.txt'), 'w') as f:
    f.write(f'CLVM training time: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)\n')
