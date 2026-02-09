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

# Load small molecule data
data_dir = os.path.join(script_dir, 'data')

background = pd.read_csv(os.path.join(data_dir, 'background_matrix.csv'), index_col=0).values
foreground = pd.read_csv(os.path.join(data_dir, 'foreground_matrix.csv'), index_col=0).values
labels_df = pd.read_csv(os.path.join(data_dir, 'labels.csv'))
foreground_label = labels_df['label'].values

target = foreground
input_dim = foreground.shape[1]

print('shape of target data:', target.shape)
print('shape of background data:', background.shape)
print('input dimension:', input_dim)

# Repeat background to match target size (required for CVAE training)
num_repeats = int(target.shape[0]/background.shape[0]) + 1
background_repeat = np.concatenate(num_repeats*[background], axis=0)
background_repeat = background_repeat[:target.shape[0]]

print('shape of repeated background:', background_repeat.shape)

start_time = time.time()
cvae, cvae_fg, z_encoder, s_encoder, cvae_decoder = contrastive_vae(input_dim=input_dim, intermediate_dim=64, latent_dim=2, disentangle=False)
history = cvae.fit([target, background_repeat], epochs=50, batch_size=100, validation_data=([target, background_repeat], None), verbose=0)
end_time = time.time()
elapsed_time = end_time - start_time
print(f'CVAE training time: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)')

z_mean, _, _ = z_encoder.predict(target, batch_size=128)

result_dir = os.path.join(script_dir, '..', 'result', 'cvae')
os.makedirs(result_dir, exist_ok=True)

result = pd.DataFrame({'label': foreground_label,
                          'latent_1': z_mean[:,0],
                          'latent_2': z_mean[:,1]})

result.to_csv(os.path.join(result_dir, 'cvae_small_mole_target2_latent.csv'), index=False)

# Save timing information
with open(os.path.join(result_dir, 'cvae_small_mole_timing.txt'), 'w') as f:
    f.write(f'CVAE training time: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)\n')
