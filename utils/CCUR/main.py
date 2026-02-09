#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import svd
from scipy.linalg import pinv


# In[3]:


def deterministic_cur(A, k, top_selected):
    U, S, Vt = svd(A, full_matrices=False)
    column_scores = compute_leverage_scores(Vt, k)
    row_scores = compute_leverage_scores(U.T, k)
    
    top_col_indices = np.argsort(-column_scores)[:top_selected]
    top_row_indices = np.argsort(-row_scores)[:top_selected]
    low_col_indices = np.argsort(column_scores)[:top_selected]

    C = A[:, top_col_indices]
    R = A[top_row_indices, :]
    C_pinv = pinv(C)
    R_pinv = pinv(R)
    U = C_pinv @ A @ R_pinv
    
    return C, column_scores, R, top_col_indices, top_row_indices, low_col_indices


def compute_leverage_scores(V, k):
    """Compute the normalized statistical leverage scores."""
    leverage_scores = np.sum(V[:k, :]**2, axis=0)
    return leverage_scores

def scores(A, k):
    U, S, Vt = svd(A, full_matrices=False)
    column_scores = compute_leverage_scores(Vt, k)
    row_scores = compute_leverage_scores(U.T, k)
    return column_scores, row_scores

def ccur(X, Y, k, num_cols, epsilon=1e-6):
    leverage_scores_X, _ = scores(X, k)
    leverage_scores_Y, _ = scores(Y, k)
    ratios = leverage_scores_X / (epsilon + leverage_scores_Y)
    top_c_indices = np.argsort(-ratios)[:num_cols]
    return top_c_indices, ratios[top_c_indices], ratios

def compute_leverage_scores(V, k):
    leverage_scores = np.sum(V[:k, :]**2, axis=0)
    return leverage_scores

def _col_row_leverage(A, k):
    U, S, Vt = svd(A, full_matrices=False)
    col_lev = compute_leverage_scores(Vt, k)   
    row_lev = compute_leverage_scores(U.T, k)  
    return col_lev, row_lev

def topk_stable(scores, k):
    order = np.lexsort((np.arange(scores.size), -scores))
    return np.sort(order[:k])

def _overlap_ratio(a_idx, b_idx, k_expected):
    if a_idx is None or b_idx is None or len(a_idx) == 0 or len(b_idx) == 0:
        return 0.0
    inter = len(np.intersect1d(a_idx, b_idx, assume_unique=False))
    return inter / float(k_expected)


def iccur(
    X, Y,
    cols=10,           # number of columns (features) to keep
    rows=20,           # number of rows (samples) to keep
    k=7,                 # rank used for leverage computations
    max_iter=10,         # hard cap on iterations
    epsilon=1e-8,        # stabilizer for column ratio
    theta_S=0.90,        # required column retention (0..1)
    theta_T=0.90,        # required row retention (0..1)
    patience=2,          # consecutive iterations meeting both thresholds
    verbose=False
):
    n_f, p = X.shape

    col_lev_Y, _ = _col_row_leverage(Y, k)

    col_lev_X_full, _ = _col_row_leverage(X, k)
    col_ratio_init = col_lev_X_full / (col_lev_Y + epsilon)
    S = topk_stable(col_ratio_init, cols)

    history = []
    prev_S = None
    prev_T = None
    prev_pair = None          
    stable_hits = 0           

    for it in range(max_iter):
        _, row_lev_on_S = _col_row_leverage(X[:, S], k)
        T = topk_stable(row_lev_on_S, min(rows, n_f))

        col_lev_X_on_T, _ = _col_row_leverage(X[T, :], k)
        col_ratio = col_lev_X_on_T / (col_lev_Y + epsilon)
        S_new = topk_stable(col_ratio, cols)

        s_overlap = _overlap_ratio(S_new, S, cols)
        t_overlap = _overlap_ratio(T, prev_T, rows) if prev_T is not None else 0.0

        avg_row = float(np.mean(row_lev_on_S[T])) if len(T) else 0.0
        avg_col = float(np.mean(col_ratio[S_new])) if len(S_new) else 0.0
        history.append({
            "iter": it,
            "col_overlap": s_overlap,
            "row_overlap": t_overlap,
            "avg_row_score": avg_row,
            "avg_col_score": avg_col
        })

  
        pair = (tuple(S_new.tolist()), tuple(T.tolist()))
        if prev_pair is not None and pair == prev_pair:
            S = S_new
            break

        if (s_overlap >= theta_S) and (t_overlap >= theta_T):
            stable_hits += 1
        else:
            stable_hits = 0

        if stable_hits >= patience:
            S = S_new
            break

        if np.array_equal(S_new, S):
            S = S_new
            break

        prev_pair = (tuple(S.tolist()), tuple(prev_T.tolist())) if prev_T is not None else None
        prev_S, prev_T = S, T
        S = S_new

    _, row_lev_on_S = _col_row_leverage(X[:, S], k)
    T = topk_stable(row_lev_on_S, min(rows, n_f))

    return S, T, history







# In[ ]:




