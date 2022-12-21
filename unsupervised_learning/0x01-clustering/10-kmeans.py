#!/usr/bin/env python3
"""k-means using sklearn."""
import sklearn.cluster


def kmeans(X, k):
    """k-means using sklearn."""
    kmeans = sklearn.cluster.KMeans(n_clusters=k).fit(X)
    return kmeans.cluster_centers_, kmeans.labels_
