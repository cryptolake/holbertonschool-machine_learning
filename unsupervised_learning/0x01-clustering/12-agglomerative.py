#!/usr/bin/env python3
"""Hierachical classification."""
import scipy.cluster.hierarchy
import matplotlib.pyplot as plt


def agglomerative(X, dist):
    """Hierachical classification."""
    Z = scipy.cluster.hierarchy.ward(X)
    plt.figure(figsize=(25, 10))
    res = scipy.cluster.hierarchy.fcluster(Z, dist, criterion='distance')
    scipy.cluster.hierarchy.dendrogram(Z, color_threshold=dist)
    plt.show()
    return res
