import os
import numpy as np
import pandas as pd
import time
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

script_dir = os.path.dirname(os.path.abspath(__file__))

# Load small molecule data
data_dir = os.path.join(script_dir, 'data')

foreground = pd.read_csv(os.path.join(data_dir, 'foreground_matrix.csv'))
foreground = foreground.set_index('Unnamed: 0')

labels = pd.read_csv(os.path.join(data_dir, 'labels.csv'))['label']

print('shape of foreground data:', foreground.shape)

# StandardScaler and PCA
scaler = StandardScaler()
X_scaled = scaler.fit_transform(np.array(foreground))

start = time.perf_counter()

pca = PCA(n_components=2)
projected_data = pca.fit_transform(X_scaled)

end = time.perf_counter()

print(f"PCA runtime: {end - start:.4f} seconds")

eigenvectors = pca.components_.T

# Save results
result_dir = os.path.join(script_dir, '..', 'result', 'pca')
os.makedirs(result_dir, exist_ok=True)

# Save projected data
result = pd.DataFrame({
    'label': labels.values,
    'latent_1': projected_data[:, 0],
    'latent_2': projected_data[:, 1]
})
result.to_csv(os.path.join(result_dir, 'pca_small_mole_latent.csv'), index=False)

# Save eigenvectors (loadings)
gene_names = foreground.columns.to_numpy()
eigenvectors_df = pd.DataFrame(eigenvectors, index=gene_names, columns=['PC1', 'PC2'])
eigenvectors_df.to_csv(os.path.join(result_dir, 'pca_small_mole_eigenvectors.csv'))

# Silhouette score
sil_score = silhouette_score(projected_data, labels, metric='euclidean')
print(f"Silhouette score: {sil_score:.4f}")

# Save timing and silhouette score
elapsed_time = end - start
with open(os.path.join(result_dir, 'pca_small_mole_timing.txt'), 'w') as f:
    f.write(f'PCA runtime: {elapsed_time:.4f} seconds\n')
    f.write(f'Silhouette score: {sil_score:.4f}\n')

print('Results saved.')
