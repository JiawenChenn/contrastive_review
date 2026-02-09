import os
import sys
import time
import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score
import contrastive_inverse_regression as cir

script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, 'data')
result_dir = os.path.join(script_dir, '..', 'result', 'cir')
os.makedirs(result_dir, exist_ok=True)

## Read in data ##
data = np.genfromtxt(os.path.join(data_dir, 'Data_Cortex_Nuclear.csv'), delimiter=',',
                     skip_header=1, usecols=range(1,78), filling_values=0)
classes = np.genfromtxt(os.path.join(data_dir, 'Data_Cortex_Nuclear.csv'), delimiter=',',
                        skip_header=1, usecols=range(78,81), dtype=None)

data_pd = pd.read_csv(os.path.join(data_dir, "Data_Cortex_Nuclear.csv"))

foreground_gene_names = data_pd.columns.values[1:-4]

## CIR ##
# CIR uses the full dataset with 8 class labels, not the binary target/background split
df = data_pd
fg = df.dropna(axis=0, how="any").copy()

# Foreground labels: map 8 classes -> 1..8 (as ints)
Y = fg["class"].astype(str).map({
    'c-CS-m': 1, 'c-CS-s': 2, 'c-SC-m': 3, 'c-SC-s': 4,
    't-CS-m': 5, 't-CS-s': 6, 't-SC-m': 7, 't-SC-s': 8
}).astype(int)
labels = np.sort(Y.unique())
L = len(labels)

feature_cols = df.columns[1:78]
X_fg_raw = fg[feature_cols].to_numpy()
n = X_fg_raw.shape[0]
X_centered = X_fg_raw - X_fg_raw.mean(axis=0, keepdims=True)

# Background: control genotype, drop NA
bg = df.loc[df["Genotype"].astype(str) == "Control"].dropna(axis=0, how="any").copy()
Yt = bg["Behavior"].astype(str).map({'C/S': 0, 'S/C': 1}).astype(int)
X_bg = bg[feature_cols].to_numpy()

# Hyperparameters
alpha = 1e-4
d = 2

# Run CIR
try:
    if hasattr(cir, "CIR"):
        start = time.perf_counter()
        V = cir.CIR(X_fg_raw, Y.to_numpy(), X_bg, Yt.to_numpy(), alpha, d)
        end = time.perf_counter()
        print(f"CIR runtime: {end - start:.4f} seconds")

        if isinstance(V, (tuple, list)):
            V = V[0]
        X_CIR = X_centered @ V

    elif hasattr(cir, "ContrastiveInverseRegression"):
        model = cir.ContrastiveInverseRegression(alpha=alpha, n_components=d)
        start = time.perf_counter()
        model.fit(X_fg_raw, Y.to_numpy(), X_bg, Yt.to_numpy())
        end = time.perf_counter()
        print(f"CIR runtime: {end - start:.4f} seconds")
        X_CIR = model.transform(X_centered)

    else:
        raise AttributeError("contrastive_inverse_regression has no CIR() or ContrastiveInverseRegression.")

except Exception as e:
    raise RuntimeError(
        f"Could not run CIR with the installed package. "
        f"Inspect the module to find the correct entry point. Original error: {e}"
    )

# Save results
result = pd.DataFrame({
    'class': Y.values,
    'latent_1': X_CIR[:, 0],
    'latent_2': X_CIR[:, 1]
})
result.to_csv(os.path.join(result_dir, 'cir_protein_latent.csv'), index=False)

# Save timing information
with open(os.path.join(result_dir, 'cir_protein_timing.txt'), 'w') as f:
    f.write(f'CIR runtime: {end - start:.4f} seconds\n')

# Silhouette score
sil_score = silhouette_score(X_CIR, Y, metric='euclidean')
print(f"Silhouette Score: {sil_score:.4f}")

with open(os.path.join(result_dir, 'cir_protein_silhouette.txt'), 'w') as f:
    f.write(f'Silhouette Score: {sil_score:.4f}\n')
