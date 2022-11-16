#!/usr/bin/env python3
"""Get mean and covariance of data set."""
import numpy as np


def mean_cov(X):
    """Get mean and covariance matrix."""
    if len(X.shape) != 2:
        raise TypeError("X must be a 2D numpy.ndarray")
    mean = np.mean(X, axis=0)
    n, d = X.shape
    if n < 2:
        raise ValueError("X must contain multiple data points")
    covm = np.zeros(shape=(d, d))
    for x in range(d):
        for y in range(d):
            covm[x, y] = np.sum((X[:, x] - mean[x])*(X[:, y] - mean[y]))/n
    return mean, covm
