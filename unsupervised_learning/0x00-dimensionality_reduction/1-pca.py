#!/usr/bin/env python3
"""principle componenet analysis."""
import numpy as np


def pca(X, ndim):
    """Perform pca while keeping features that cover at least var variance."""
    X = X - X.mean(axis=0)
    _, _, v = np.linalg.svd(X)
    W = v[:ndim].T
    return X @ W
