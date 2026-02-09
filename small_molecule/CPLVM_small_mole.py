import os
import numpy as np
import pandas as pd
import time
from cplvm import CPLVM, CPLVMLogNormalApprox
from sklearn.metrics import silhouette_score

script_dir = os.path.dirname(os.path.abspath(__file__))

# Load small molecule count data
data_dir = os.path.join(script_dir, 'data')

background = pd.read_csv(os.path.join(data_dir, 'count', 'background_count.csv'), index_col=0)
foreground = pd.read_csv(os.path.join(data_dir, 'count', 'foreground_count.csv'), index_col=0)
labels_df = pd.read_csv(os.path.join(data_dir, 'labels.csv'))
foreground_label = labels_df['label'].values

print('shape of foreground (target) data:', foreground.shape)
print('shape of background data:', background.shape)

# In CPLVM: X = background (control), Y = foreground (target)
# ty captures foreground-specific variation in Y
X = background.T   # genes x samples
Y = foreground.T   # genes x samples

k_shared = 2
k_foreground = 2

start_time = time.time()
cplvm_model = CPLVM(
    k_shared=k_shared,
    k_foreground=k_foreground,
    compute_size_factors=True,
    offset_term=False
)

approx_model = CPLVMLogNormalApprox(
    X, Y,
    k_shared=k_shared,
    k_foreground=k_foreground,
    compute_size_factors=True,
    offset_term=False
)

print('Fitting CPLVM model...')
model_output = cplvm_model.fit_model_vi(X, Y, approximate_model=approx_model)
end_time = time.time()
elapsed_time = end_time - start_time
print(f'CPLVM training time: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)')

am = model_output["approximate_model"]

# Mean of log-normal (take mean of posterior)
S_estimated = np.exp(am.qs_mean.numpy() + am.qs_stddv.numpy() ** 2 / 2)
W_estimated = np.exp(am.qw_mean.numpy() + am.qw_stddv.numpy() ** 2 / 2)
zx_estimated = np.exp(am.qzx_mean.numpy() + am.qzx_stddv.numpy() ** 2 / 2)
zy_estimated = np.exp(am.qzy_mean.numpy() + am.qzy_stddv.numpy() ** 2 / 2)
ty_estimated = np.exp(am.qty_mean.numpy() + am.qty_stddv.numpy() ** 2 / 2)

print('shape of S (shared loading):', S_estimated.shape)
print('shape of W (foreground loading):', W_estimated.shape)
print('shape of zx (shared latent, background):', zx_estimated.shape)
print('shape of zy (shared latent, foreground):', zy_estimated.shape)
print('shape of ty (foreground-specific latent):', ty_estimated.shape)

result_dir = os.path.join(script_dir, '..', 'result', 'cplvm')
os.makedirs(result_dir, exist_ok=True)

# Save loading matrices
S_df = pd.DataFrame(S_estimated, index=foreground.columns.values)
W_df = pd.DataFrame(W_estimated, index=foreground.columns.values)
S_df.to_csv(os.path.join(result_dir, 'cplvm_small_mole_loading_s.csv'))
W_df.to_csv(os.path.join(result_dir, 'cplvm_small_mole_loading_w.csv'))

# Foreground-specific latent representation (ty)
ty_df = pd.DataFrame(ty_estimated.T, columns=['latent_1', 'latent_2'])
ty_df.insert(0, 'label', foreground_label)

ty_df.to_csv(os.path.join(result_dir, 'cplvm_small_mole_shared2_target2_latent.csv'), index=False)
print('Results saved.')

# Silhouette score
score = silhouette_score(X=ty_estimated.T, labels=foreground_label)
print(f'Silhouette score: {score:.4f}')

# Save timing information
with open(os.path.join(result_dir, 'cplvm_small_mole_timing.txt'), 'w') as f:
    f.write(f'CPLVM training time: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)\n')
    f.write(f'Silhouette score: {score:.4f}\n')
