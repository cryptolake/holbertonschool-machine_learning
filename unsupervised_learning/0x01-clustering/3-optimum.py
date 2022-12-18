#!/usr/bin/env python3
"""Optimum number of clusters."""
import numpy as np

kmeans = __import__('1-kmeans').kmeans
variance = __import__('2-variance').variance


def optimum_k(X, kmin=1, kmax=None, iterations=1000):
    """Optimum number of clusters."""
    if type(X) is not np.ndarray or len(X.shape) != 2:
        return None, None

    if type(kmin) is not int or kmin < 1:
        return None, None

    if kmax is None:
        kmax = X.shape[0]

    if type(kmax) is not int or kmax < 1:
        return None, None

    if type(iterations) is not int or iterations < 1:
        return None, None

    if kmax - kmin < 2:
        return None, None

    d_vars = []
    results = []
    for k in range(kmin, kmax+1):
        k_m = kmeans(X, k, iterations)
        results.append(k_m)
        d_vars.append(variance(X, k_m[0]))
    d_vars = np.array(d_vars)
    d_vars = list(abs(d_vars - d_vars[0]))
    return results, d_vars
