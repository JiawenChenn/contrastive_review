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


background = np.load(os.path.join(script_dir, 'data', 'background.npy'))
foreground = np.load(os.path.join(script_dir, 'data', 'foreground.npy'))
foreground_label = np.load(os.path.join(script_dir, 'data', 'foreground_labels.npy'))

data = np.concatenate([background, foreground], axis=0)
background_idx = np.arange(0, background.shape[0])
target_idx = np.arange(background.shape[0], data.shape[0])

target = foreground

print('shape of target data:', target.shape)

start_time = time.time()
cvae, cvae_fg, z_encoder, s_encoder, cvae_decoder = contrastive_vae(input_dim=784, intermediate_dim=64, latent_dim=2, disentangle=False)
history = cvae.fit([target, background], epochs=50, batch_size=100, validation_data=([target, background], None), verbose=0)
end_time = time.time()
elapsed_time = end_time - start_time
print(f'CVAE training time: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)')

z_mean, _, _ = z_encoder.predict(target, batch_size=128)

result_dir = os.path.join(script_dir, '..', 'result', 'cvae')
os.makedirs(result_dir, exist_ok=True)

result = pd.DataFrame({'class': foreground_label,
                          'latent_1': z_mean[:,0],
                          'latent_2': z_mean[:,1]})

result.to_csv(os.path.join(result_dir, 'cvae_mnist_target2_latent.csv'))

# Save timing information
with open(os.path.join(result_dir, 'cvae_mnist_timing.txt'), 'w') as f:
    f.write(f'CVAE training time: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)\n')
