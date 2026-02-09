import os
import sys
import time
import numpy as np
import pandas as pd
from scipy.linalg import pinv
from scipy.stats import norm as norm_dist
from scipy.stats import chi2
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, 'data')
result_dir = os.path.join(script_dir, '..', 'result', 'bascod')
os.makedirs(result_dir, exist_ok=True)

## Read in data ##
data = np.genfromtxt(os.path.join(data_dir, 'Data_Cortex_Nuclear.csv'), delimiter=',',
                     skip_header=1, usecols=range(1,78), filling_values=0)
classes = np.genfromtxt(os.path.join(data_dir, 'Data_Cortex_Nuclear.csv'), delimiter=',',
                        skip_header=1, usecols=range(78,81), dtype=None)

data_pd = pd.read_csv(os.path.join(data_dir, "Data_Cortex_Nuclear.csv"))

# Handle both string and bytes comparisons from genfromtxt dtype=None
sample_val = classes[0, -1]
if isinstance(sample_val, bytes):
    SC, CS, Saline, Control, Ts65Dn = b'S/C', b'C/S', b'Saline', b'Control', b'Ts65Dn'
else:
    SC, CS, Saline, Control, Ts65Dn = 'S/C', 'C/S', 'Saline', 'Control', 'Ts65Dn'

target_idx_A = np.where((classes[:,-1]==SC) & (classes[:,-2]==Saline) & (classes[:,-3]==Control))[0]
target_idx_B = np.where((classes[:,-1]==SC) & (classes[:,-2]==Saline) & (classes[:,-3]==Ts65Dn))[0]

labels = len(target_idx_A)*[0] + len(target_idx_B)*[1]
target_idx = np.concatenate((target_idx_A, target_idx_B))

target = data[target_idx]

background_idx = np.where((classes[:,-1]==CS) & (classes[:,-2]==Saline) & (classes[:,-3]==Control))
background = data[background_idx]

foreground_gene_names = data_pd.columns.values[1:-4]

## BasCoD ##
def lm_no_intercept(X, E):
    coef = np.linalg.lstsq(E, X, rcond=None)[0]
    return coef.T

def BasCoD_single(X_0, X_j, R_0, R_j, embedding_0, embedding_j, eps=0.3):
    p = X_0.shape[1]

    embedding_0 = embedding_0[:, :R_0]
    embedding_j = embedding_j[:, :R_j]

    Gamma0hat = lm_no_intercept(X_0, embedding_0)  # (p, R_0)
    Gammajhat = lm_no_intercept(X_j, embedding_j)  # (p, R_j)

    Gammajhat = Gammajhat / np.linalg.norm(Gammajhat, axis=0, keepdims=True)

    G0_pinv = pinv(Gamma0hat)
    proj = Gamma0hat @ G0_pinv @ Gammajhat  # (p, R_j)

    if R_j != 1:
        rho = np.diag(np.corrcoef(Gammajhat.T, proj.T)[:R_j, R_j:])
    else:
        rho = np.corrcoef(Gammajhat.ravel(), proj.ravel())[0, 1]

    # Test statistic
    z = (np.arctanh(rho) - np.arctanh(1 - eps)) * np.sqrt(p - 3)
    stat = np.sum(-2 * np.log(norm_dist.cdf(z)))

    pval = chi2.sf(stat, df=2 * R_j)

    print(f"BasCoD p-value is : {pval}")

    return {
        "corr_vals": rho,
        "pvalue": pval
    }

# Standardize
X_0 = np.nan_to_num(target, nan=0.0)
X_0 = StandardScaler(with_mean=True, with_std=True).fit_transform(X_0)

X_j = np.nan_to_num(background, nan=0.0)
X_j = StandardScaler(with_mean=True, with_std=True).fit_transform(X_j)

# PCA embeddings
R_0 = 10
R_j = 7

pca_0 = PCA()
pca_j = PCA()

embedding_0 = pca_0.fit_transform(X_0)
embedding_j = pca_j.fit_transform(X_j)

start = time.perf_counter()

bascod_res = BasCoD_single(
    X_0=X_0,
    X_j=X_j,
    R_0=R_0,
    R_j=R_j,
    embedding_0=embedding_0,
    embedding_j=embedding_j,
    eps=0.3
)

end = time.perf_counter()
print(f"BasCoD runtime: {end - start:.4f} seconds")

# Save results
results_df = pd.DataFrame({
    'pvalue': [bascod_res['pvalue']],
    'corr_vals': [str(bascod_res['corr_vals'])]
})
results_df.to_csv(os.path.join(result_dir, 'bascod_protein_results.csv'), index=False)

# Save timing information
with open(os.path.join(result_dir, 'bascod_protein_timing.txt'), 'w') as f:
    f.write(f'BasCoD runtime: {end - start:.4f} seconds\n')
    f.write(f'BasCoD p-value: {bascod_res["pvalue"]}\n')
