import os
import numpy as np
import pandas as pd
import time
from sklearn.metrics import silhouette_score

script_dir = os.path.dirname(os.path.abspath(__file__))

# Load small molecule data
data_dir = os.path.join(script_dir, 'data')

foreground = pd.read_csv(os.path.join(data_dir, 'foreground_matrix.csv'))
background = pd.read_csv(os.path.join(data_dir, 'background_matrix.csv'))
labels = pd.read_csv(os.path.join(data_dir, 'labels.csv'))['label']

foreground = foreground.set_index('Unnamed: 0')
background = background.set_index('Unnamed: 0')

print('shape of foreground data:', foreground.shape)
print('shape of background data:', background.shape)

# Center foreground and background
foreground_centered = (foreground - np.mean(foreground, axis=0))
background_centered = (background - np.mean(background, axis=0))

start = time.perf_counter()

# Compute covariance matrices and contrastive covariance
cov_foreground = np.cov(foreground_centered, rowvar=False)
cov_background = np.cov(background_centered, rowvar=False)

alpha = 1
cov_diff = cov_foreground - alpha * cov_background

# Eigen decomposition
eigenvalues, eigenvectors = np.linalg.eig(cov_diff)

end = time.perf_counter()

print(f"CPCA runtime: {end - start:.4f} seconds")

# Sort by descending eigenvalue and project
idx = eigenvalues.argsort()[::-1]
eigenvectors = eigenvectors[:, idx]
projected_data = foreground_centered @ eigenvectors[:, :2]
projected_data = projected_data.to_numpy()

# Save results
result_dir = os.path.join(script_dir, '..', 'result', 'cpca')
os.makedirs(result_dir, exist_ok=True)

# Save projected data
result = pd.DataFrame({
    'label': labels.values,
    'latent_1': projected_data[:, 0],
    'latent_2': projected_data[:, 1]
})
result.to_csv(os.path.join(result_dir, 'cpca_small_mole_latent.csv'), index=False)

# Save eigenvectors (loadings)
gene_names = foreground.columns.to_numpy()
eigenvectors_df = pd.DataFrame(
    eigenvectors[:, :2],
    index=gene_names,
    columns=['PC1', 'PC2']
)
eigenvectors_df.to_csv(os.path.join(result_dir, 'cpca_small_mole_eigenvectors.csv'))

# Silhouette score
sil_score = silhouette_score(projected_data, labels, metric='euclidean')
print(f"Silhouette score: {sil_score:.4f}")

# Save timing and silhouette score
elapsed_time = end - start
with open(os.path.join(result_dir, 'cpca_small_mole_timing.txt'), 'w') as f:
    f.write(f'CPCA runtime: {elapsed_time:.4f} seconds\n')
    f.write(f'Silhouette score: {sil_score:.4f}\n')

print('Results saved.')
