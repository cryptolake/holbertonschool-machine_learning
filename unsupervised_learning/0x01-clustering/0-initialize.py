#!/usr/bin/env python3
"""Initialize k-means clustering."""
import numpy as np


def initialize(X, k):
    """Initialize k-means clustering."""
    if type(k) is not int or k <= 0:
        return None
    mins = np.amin(X, axis=0)
    maxes = np.amax(X, axis=0)
    centroids = np.random.uniform(low=mins, high=maxes, size=(k, X.shape[1]))
    return centroids
