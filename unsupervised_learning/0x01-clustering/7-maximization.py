#!/usr/bin/env python3
"""Maximization step in EM algorithm for GMM."""
import numpy as np


def maximization(X, g):
    """Perform Maximization step in EM algorithm for GMM."""
    if type(X) is not np.ndarray or len(X.shape) != 2:
        return None, None
    n, d = X.shape
    n, _ = X.shape
    k, _ = g.shape
    pi = []
    m = []
    S = []
    for j in range(k):
        yj = np.expand_dims(g[j], axis=1)
        pi.append(yj.sum()/n)
        m.append(np.expand_dims(np.sum(yj * X, axis=0),
                                axis=0)/np.sum(yj))
        diff = (X - m[j])
        weighted_sum = (g[j] * diff.T) @ diff
        S.append(np.expand_dims(weighted_sum / np.sum(yj), axis=0))
    m = np.concatenate(m, axis=0)
    pi = np.array(pi)
    S = np.concatenate(S, axis=0)
    return pi, m, S
