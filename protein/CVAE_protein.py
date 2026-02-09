import os
import sys
import time

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(script_dir, '..', 'utils', 'contrastive_vae'))

import tensorflow as tf
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from vae_utils import generate_data, standard_vae, contrastive_vae, plot_latent_space, plot_latent_space4d, contrastive_vae_no_bias, plot_clean_digits_only, plot_sweeps_mnist
from sklearn.metrics import silhouette_score

import numpy as np, h5py
import matplotlib.pyplot as plt
from operator import itemgetter
from scipy.sparse import vstack, coo_matrix, csc_matrix, isspmatrix_csc
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

num_repeats = int(target.shape[0]/background.shape[0]) + 1
background_repeat = np.concatenate(num_repeats*[background], axis=0)
background_repeat = background_repeat[:target.shape[0]]

start_time = time.time()
cvae, cvae_fg, z_encoder, s_encoder, cvae_decoder = contrastive_vae(input_dim=77, intermediate_dim=64, latent_dim=2, disentangle=False)
history = cvae.fit([target, background_repeat], epochs=50, batch_size=100, validation_data=([target, background_repeat], None), verbose=0)
end_time = time.time()
elapsed_time = end_time - start_time
print(f'CVAE training time: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)')

z_mean, _, _ = z_encoder.predict(target, batch_size=128)

result_dir = os.path.join(script_dir, '..', 'result', 'cvae')
os.makedirs(result_dir, exist_ok=True)

result = pd.DataFrame({'class1': classes[target_idx,-3],
                          'class2': classes[target_idx,-2],
                            'class3': classes[target_idx,-1],
                          'latent_1': z_mean[:,0],
                          'latent_2': z_mean[:,1]})

result.to_csv(os.path.join(result_dir, 'cvae_protein_target2_latent.csv'))

# Save timing information
with open(os.path.join(result_dir, 'cvae_protein_timing.txt'), 'w') as f:
    f.write(f'CVAE training time: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)\n')
