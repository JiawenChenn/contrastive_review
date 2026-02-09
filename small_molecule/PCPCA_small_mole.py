import os
import numpy as np
import pandas as pd
import time
from pcpca import PCPCA
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

# Fit PCPCA
start = time.perf_counter()

pcpca = PCPCA(gamma=0.5, n_components=2)
pcpca.fit(foreground_centered.T, background_centered.T)

end = time.perf_counter()
print(f"PCPCA runtime: {end - start:.4f} seconds")

# Get loadings
W = pcpca.W_mle

# Transform data
X_reduced, Y_reduced = pcpca.fit_transform(
    foreground_centered.T,
    background_centered.T
)

projected_data = np.array(X_reduced).T

# Save results
result_dir = os.path.join(script_dir, '..', 'result', 'pcpca')
os.makedirs(result_dir, exist_ok=True)

# Save W_mle loadings
gene_names = foreground.columns.to_numpy()
W_df = pd.DataFrame(W, index=gene_names, columns=['W1', 'W2'])
W_df.to_csv(os.path.join(result_dir, 'pcpca_small_mole_W_mle.csv'))

# Save projected data
result = pd.DataFrame({
    'label': labels.values,
    'latent_1': projected_data[:, 0],
    'latent_2': projected_data[:, 1]
})
result.to_csv(os.path.join(result_dir, 'pcpca_small_mole_latent.csv'), index=False)

# Silhouette score
sil_score = silhouette_score(
    projected_data[:, :2],
    labels,
    metric='euclidean'
)
print(f"Silhouette score: {sil_score:.4f}")

# Save timing and silhouette score
elapsed_time = end - start
with open(os.path.join(result_dir, 'pcpca_small_mole_timing.txt'), 'w') as f:
    f.write(f'PCPCA runtime: {elapsed_time:.4f} seconds\n')
    f.write(f'Silhouette score: {sil_score:.4f}\n')

print('Results saved.')
