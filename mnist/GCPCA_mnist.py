import os
import numpy as np
import pandas as pd
import time
from sklearn.metrics import silhouette_score
import generalized_contrastive_PCA as gcPCA

script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, 'data')
result_dir = os.path.join(script_dir, '..', 'result', 'gcpca')
os.makedirs(result_dir, exist_ok=True)

# Load data
foreground = np.load(os.path.join(data_dir, 'foreground.npy'))
background = np.load(os.path.join(data_dir, 'background.npy'))
labels = np.load(os.path.join(data_dir, 'foreground_labels.npy'))

# Convert to DataFrame (as done in notebook)
foreground = pd.DataFrame(foreground)
background = pd.DataFrame(background)

# GCPCA: fit with method v4.1
Ra = np.array(foreground.copy())
Rb = np.array(background.copy())

start = time.perf_counter()

model = gcPCA.gcPCA(method='v4.1', normalize_flag=True)
model.fit(Ra, Rb)

end = time.perf_counter()
elapsed_time = end - start
print(f"GCPCA runtime: {elapsed_time:.4f} seconds")

# Extract top 2 components from Ra_scores_
Z_a = model.Ra_scores_[:, :2]

# Compute silhouette score
labels = np.asarray(labels).ravel()
cluster_labels = labels.astype(int)
sil_score = silhouette_score(Z_a, cluster_labels, metric='euclidean')
print(f"SS: {sil_score:.4f}")

# Save latent representations with labels
result = pd.DataFrame({
    'classes': labels,
    'latent_1': Z_a[:, 0],
    'latent_2': Z_a[:, 1]
})
result.to_csv(os.path.join(result_dir, 'gcpca_mnist_latent.csv'), index=False)

# Save timing information
with open(os.path.join(result_dir, 'gcpca_mnist_timing.txt'), 'w') as f:
    f.write(f'GCPCA runtime: {elapsed_time:.4f} seconds\n')
    f.write(f'Silhouette Score: {sil_score:.4f}\n')
