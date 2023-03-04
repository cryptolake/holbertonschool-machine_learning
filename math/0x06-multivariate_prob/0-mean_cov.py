#!/usr/bin/env python3
"""Get mean and covariance of data set."""
import numpy as np


def mean_cov(X):
    """Get mean and covariance matrix."""
    if type(X) is not np.ndarray or len(X.shape) != 2:
        raise TypeError("X must be a 2D numpy.ndarray")
    mean = np.mean(X, axis=0, keepdims=True)
    # n, d = X.shape
    n, _ = X.shape
    if n < 2:
        raise ValueError("X must contain multiple data points")
    # Dot products measure similarity, and here we are measuring the similiraty
    # between features.
    return mean, np.matmul((X - mean).T, (X - mean)) / (X.shape[0] - 1)
