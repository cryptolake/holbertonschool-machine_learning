#!/usr/bin/env python3
"""Initialize k-means clustering."""
import numpy as np


def kmeans(X, k, iterations=1000):
    """Perform k-means clustering."""
    d = X.shape[1]
    clustroids = X[:k]
    if clustroids is None:
        return None
    clss = np.ndarray(shape=(X.shape[0], ))

    for _ in range(iterations):
        clus = X[..., np.newaxis] - clustroids.T
        clss = np.argmin(np.linalg.norm(clus, axis=1), axis=1)
        n_clustroids = np.ndarray(shape=(k, d))
        for km in range(k):
            n_clustroids[km, :] = np.mean(X[clss == km, :], axis=0)
        clustroids = n_clustroids
    return clustroids, clss
