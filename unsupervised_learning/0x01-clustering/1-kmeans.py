#!/usr/bin/env python3
"""Initialize k-means clustering."""
import numpy as np


def initialize(X, k):
    """Initialize k-means clustering."""
    if type(k) is not int or k <= 0:
        return None
    if type(X) is not np.ndarray or len(X.shape) != 2:
        return None
    mins = np.min(X, axis=0)
    maxes = np.max(X, axis=0)
    centroids = np.random.uniform(low=mins, high=maxes, size=(k, X.shape[1]))
    return centroids


def get_clss(X, clustroids):
    """Get clss."""
    clus = X - clustroids[:, np.newaxis]
    clss = np.argmin(np.sqrt(np.sum(clus**2, axis=2)), axis=0)
    return clss


def kmeans(X, k, iterations=1000):
    """Perform k-means clustering."""
    clustroids = initialize(X, k)
    if clustroids is None:
        return None, None
    if type(iterations) is not int or iterations < 1:
        return None, None

    for _ in range(iterations):
        clss = get_clss(X, clustroids)
        n_clustroids = clustroids.copy()
        for km in range(k):
            if (X[clss == km].size == 0):
                n_clustroids[km] = initialize(X, 1)
            else:
                n_clustroids[km] = np.mean(X[clss == km], axis=0)
        if np.all(n_clustroids == clustroids):
            break
        clustroids = n_clustroids
    return clustroids, get_clss(X, clustroids)
