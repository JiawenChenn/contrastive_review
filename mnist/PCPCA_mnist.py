import os
import numpy as np
import pandas as pd
import time
from sklearn.metrics import silhouette_score
from pcpca import PCPCA

script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, 'data')
result_dir = os.path.join(script_dir, '..', 'result', 'pcpca')
os.makedirs(result_dir, exist_ok=True)

# Load data
foreground = np.load(os.path.join(data_dir, 'foreground.npy'))
background = np.load(os.path.join(data_dir, 'background.npy'))
labels = np.load(os.path.join(data_dir, 'foreground_labels.npy'))

# Convert to DataFrame (as done in notebook)
foreground = pd.DataFrame(foreground)
background = pd.DataFrame(background)

# PCPCA: fit and transform
start = time.perf_counter()

pcpca = PCPCA(gamma=0.9, n_components=2)
pcpca.fit(foreground.T, background.T)

end = time.perf_counter()
elapsed_time = end - start
print(f"PCPCA runtime: {elapsed_time:.4f} seconds")

# Transform
X_reduced, Y_reduced = pcpca.fit_transform(foreground.T, background.T)
X_reduced = np.array(X_reduced)

# Compute silhouette score
labels = np.asarray(labels).ravel()
cluster_labels = labels.astype(int)
sil_score = silhouette_score(X_reduced.T, cluster_labels, metric='euclidean')
print(f"SS: {sil_score:.4f}")

# Save latent representations with labels
result = pd.DataFrame({
    'classes': labels,
    'latent_1': X_reduced.T[:, 0],
    'latent_2': X_reduced.T[:, 1]
})
result.to_csv(os.path.join(result_dir, 'pcpca_mnist_latent.csv'), index=False)

# Save timing information
with open(os.path.join(result_dir, 'pcpca_mnist_timing.txt'), 'w') as f:
    f.write(f'PCPCA runtime: {elapsed_time:.4f} seconds\n')
    f.write(f'Silhouette Score: {sil_score:.4f}\n')
