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



data = np.genfromtxt(os.path.join(script_dir, 'data', 'Data_Cortex_Nuclear.csv'), delimiter=',',
                     skip_header=1, usecols=range(1,78), filling_values=0)
classes = np.genfromtxt(os.path.join(script_dir, 'data', 'Data_Cortex_Nuclear.csv'), delimiter=',',
                        skip_header=1, usecols=range(78,81), dtype=None)

target_idx_A = np.where((classes[:,-1]==b'S/C') & (classes[:,-2]==b'Saline') & (classes[:,-3]==b'Control'))[0]
target_idx_B = np.where((classes[:,-1]==b'S/C') & (classes[:,-2]==b'Saline') & (classes[:,-3]==b'Ts65Dn'))[0]

labels = len(target_idx_A)*[0] + len(target_idx_B)*[1]
target_idx = np.concatenate((target_idx_A,target_idx_B))

target = data[target_idx]

background_idx = np.where((classes[:,-1]==b'C/S') & (classes[:,-2]==b'Saline') & (classes[:,-3]==b'Control'))
background = data[background_idx]


print('shape of target data:', target.shape)
print('shape of background data:', background.shape)

result_dir = os.path.join(script_dir, '..', 'result', 'clvm')
os.makedirs(result_dir, exist_ok=True)

start_time = time.time()
model_0 = clvm(target, background, k_shared=2, k_target=2)
t_clvm = model_0.variational_inference(num_epochs=10000, plot=False, fp=os.path.join(result_dir, 'protein_shared2_target2_'))
end_time = time.time()
elapsed_time = end_time - start_time
print(f'CLVM training time: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)')

np.savetxt(os.path.join(result_dir, 'clvm_protein_shared2_target2_loading_w.txt'), model_0.w_hat)


result = pd.DataFrame({'classes': classes[target_idx,-3],
                       'latent_1': t_clvm[:,0],
                       'latent_2': t_clvm[:,1]})

result.to_csv(os.path.join(result_dir, 'clvm_protein_shared2_target2_latent.csv'), index=False)

# Save timing information
with open(os.path.join(result_dir, 'clvm_protein_timing.txt'), 'w') as f:
    f.write(f'CLVM training time: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)\n')
