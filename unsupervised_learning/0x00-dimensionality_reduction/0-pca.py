#!/usr/bin/env python3
"""Simple principle componenet analysis."""
import numpy as np


def pca(X, var=0.95):
    """Perform pca while keeping features that cover at least var variance."""
    X = X / X.std()
    cov = X.T @ X
    eig_val, eig_vec = np.linalg.eig(cov)
    oi = np.argsort(eig_val)[::-1]
    sorted_eigenvectors = eig_vec[:, oi]
    explained = [i / sum(eig_val) for i in eig_val]
    explained = np.cumsum(explained)
    arg = np.argwhere(explained >= var)[0, 0]
    return -1*sorted_eigenvectors[:, 0:arg]
