import os
import numpy as np
import pandas as pd
import time
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from numpy.linalg import pinv, norm
from scipy.stats import norm as norm_dist
from scipy.stats import chi2

script_dir = os.path.dirname(os.path.abspath(__file__))

# Load small molecule data
data_dir = os.path.join(script_dir, 'data')

foreground = pd.read_csv(os.path.join(data_dir, 'foreground_matrix.csv'))
background = pd.read_csv(os.path.join(data_dir, 'background_matrix.csv'))

foreground = foreground.set_index('Unnamed: 0')
background = background.set_index('Unnamed: 0')

print('shape of foreground data:', foreground.shape)
print('shape of background data:', background.shape)


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

    z = (np.arctanh(rho) - np.arctanh(1 - eps)) * np.sqrt(p - 3)
    stat = np.sum(-2 * np.log(norm_dist.cdf(z)))

    pval = chi2.sf(stat, df=2 * R_j)

    print(f"BasCoD p-value is : {pval}")

    return {
        "corr_vals": rho,
        "pvalue": pval
    }


# Standardize foreground and background
X_0 = np.nan_to_num(foreground, nan=0.0)
X_0 = StandardScaler(with_mean=True, with_std=True).fit_transform(X_0)

X_j = np.nan_to_num(background, nan=0.0)
X_j = StandardScaler(with_mean=True, with_std=True).fit_transform(X_j)

# PCA embeddings
R_0 = 8
R_j = 5

pca_0 = PCA()
pca_j = PCA()

start = time.perf_counter()

embedding_0 = pca_0.fit_transform(X_0)
embedding_j = pca_j.fit_transform(X_j)

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
result_dir = os.path.join(script_dir, '..', 'result', 'bascod')
os.makedirs(result_dir, exist_ok=True)

result = pd.DataFrame({
    'corr_vals': [bascod_res['corr_vals'].tolist() if hasattr(bascod_res['corr_vals'], 'tolist') else bascod_res['corr_vals']],
    'pvalue': [bascod_res['pvalue']]
})
result.to_csv(os.path.join(result_dir, 'bascod_small_mole_result.csv'), index=False)

# Save timing
elapsed_time = end - start
with open(os.path.join(result_dir, 'bascod_small_mole_timing.txt'), 'w') as f:
    f.write(f'BasCoD runtime: {elapsed_time:.4f} seconds\n')
    f.write(f'BasCoD p-value: {bascod_res["pvalue"]}\n')
    f.write(f'Correlation values: {bascod_res["corr_vals"]}\n')

print('Results saved.')
