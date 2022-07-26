#!/usr/bin/env python3
"""Maximization step in EM algorithm for GMM."""
import numpy as np


def maximization(X, g):
    """Perform Maximization step in EM algorithm for GMM."""
    # https://python-course.eu/machine-learning/expectation-maximization-and-gaussian-mixture-models-gmm.php
    if type(X) is not np.ndarray or len(X.shape) != 2:
        return None, None, None
    n, _ = X.shape
    if type(g) is not np.ndarray or len(g.shape) != 2 or g.shape[1] != n or \
            not np.isclose(g.sum(), n):
        return None, None, None
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
