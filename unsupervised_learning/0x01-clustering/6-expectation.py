#!/usr/bin/env python3
"""Expectation step in EM algorithm for GMM."""
import numpy as np

pdf = __import__('5-pdf').pdf


def expectation(X, pi, m, S):
    """Perform Expectation step in EM algorithm for GMM."""
    if type(X) is not np.ndarray or len(X.shape) != 2:
        return None, None
    n, d = X.shape
    if type(pi) is not np.ndarray or len(pi.shape) != 1 or\
            not np.isclose(np.sum(pi), 1):
        return None, None
    k_d = pi.shape[0]
    if type(m) is not np.ndarray or len(m.shape) != 2\
            or m.shape != (k_d, d):
        return None, None
    if type(S) is not np.ndarray or len(S.shape) != 3\
            or S.shape != (k_d, d, d):
        return None, None

    G = np.ndarray(shape=(k_d, n))
    for k in range(k_d):
        G[k] = pi[k]*pdf(X, m[k], S[k])
    sum_k = G.sum(axis=0)
    res = G / sum_k
    return res, np.sum(np.log(sum_k))
