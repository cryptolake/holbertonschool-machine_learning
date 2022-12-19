#!/usr/bin/env python3
"""Initialize GMM."""
import numpy as np

kmeans = __import__('1-kmeans').kmeans


def initialize(X, k):
    """Initialize GMM."""
    if type(X) is not np.ndarray or len(X.shape) != 2:
        return None, None, None
    if type(k) is not int or k <= 0:
        return None, None, None

    _, d = X.shape
    m, _ = kmeans(X, k)
    pi = np.ones(shape=(k)) / k
    S = np.concatenate([np.expand_dims(np.identity(d), axis=0)]*k, axis=0)
    return pi, m, S
