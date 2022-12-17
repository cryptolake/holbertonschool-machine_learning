#!/usr/bin/env python3
"""Initialize k-means clustering."""
import numpy as np


def initialize(X, k):
    """Initialize k-means clustering."""
    if type(k) is not int or k <= 0:
        return None
    if type(X) is not np.ndarray or len(X.shape) != 2:
        return None
    mins = np.amin(X, axis=0)
    maxes = np.amax(X, axis=0)
    centroids = np.random.uniform(low=mins, high=maxes, size=(k, X.shape[1]))
    return centroids


def kmeans(X, k, iterations=1000):
    """Perform k-means clustering."""
    d = X.shape[1]
    clustroids = initialize(X, k)
    if clustroids is None:
        return None
    clss = np.ndarray(shape=(X.shape[0], ))

    for _ in range(iterations):
        # TODO: use this loop for everthing by going one clustroid at a time
        clusters = [
            np.expand_dims(abs(X-clustroids[km]), 2) for km in range(k)
        ]
        clusters = np.concatenate(clusters, axis=2)
        clusters = np.sum(clusters, axis=1)
        clss = np.argmin(clusters, axis=1)
        n_clustroids = np.ndarray(shape=(k, d))
        for km in range(k):
            n_clustroids[km, :] = np.mean(X[clss == km, :], axis=0)

        if (n_clustroids == clustroids).all():
            return clustroids, clss
        clustroids = n_clustroids
    return clustroids, clss
