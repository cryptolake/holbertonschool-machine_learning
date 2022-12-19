#!/usr/bin/env python3
"""PDF of the gaussian distribution (multivariate)."""
import numpy as np


def pdf(X, m, S):
    """PDF of the gaussian distribution (multivariate)."""
    if type(X) is not np.ndarray or len(X.shape) != 2:
        return None

    _, d = X.shape

    if type(m) is not np.ndarray or len(m.shape) != 1 or m.shape[0] != d:
        return None
    if type(S) is not np.ndarray or len(S.shape) != 2 or S.shape != (d, d):
        return None

    x_m = X - m
    det = np.linalg.det(S)
    inv = np.linalg.inv(S)
    num = np.exp(-0.5 * np.sum(x_m * np.matmul(inv, x_m.T).T, axis=1))
    denum = np.sqrt(((2*np.pi)**d)*det)
    return np.maximum(num/denum, 1e-300)
