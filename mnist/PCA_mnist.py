import os
import numpy as np
import pandas as pd
import time
from sklearn.metrics import silhouette_score

script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, 'data')
result_dir = os.path.join(script_dir, '..', 'result', 'pca')
os.makedirs(result_dir, exist_ok=True)

# Load data
foreground = np.load(os.path.join(data_dir, 'foreground.npy'))
background = np.load(os.path.join(data_dir, 'background.npy'))
labels = np.load(os.path.join(data_dir, 'foreground_labels.npy'))

# Convert to DataFrame (as done in notebook)
foreground = pd.DataFrame(foreground)
background = pd.DataFrame(background)

def cov_rows(X, center=True):
    X = np.asarray(X)
    Xc = X - X.mean(axis=0, keepdims=True) if center else X
    return (Xc.T @ Xc) / Xc.shape[0]

# PCA via eigendecomposition of covariance matrix
start = time.perf_counter()

Sf = cov_rows(foreground, center=False)

vals, vecs = np.linalg.eig(Sf)

end = time.perf_counter()
elapsed_time = end - start
print(f"PCA runtime: {elapsed_time:.4f} seconds")

# Select top 2 eigenvectors
order = np.argsort(vals.real)[::-1]
W = vecs[:, order[:2]].real

# Project foreground data
Z = foreground @ W  # (n_f x 2)

# Compute silhouette score
labels = np.asarray(labels).ravel()
cluster_labels = labels.astype(int)
sil_score = silhouette_score(Z, cluster_labels, metric='euclidean')
print(f"SS: {sil_score:.4f}")

# Save latent representations with labels
result = pd.DataFrame({
    'classes': labels,
    'latent_1': Z.iloc[:, 0],
    'latent_2': Z.iloc[:, 1]
})
result.to_csv(os.path.join(result_dir, 'pca_mnist_latent.csv'), index=False)

# Save timing information
with open(os.path.join(result_dir, 'pca_mnist_timing.txt'), 'w') as f:
    f.write(f'PCA runtime: {elapsed_time:.4f} seconds\n')
    f.write(f'Silhouette Score: {sil_score:.4f}\n')
