#!/usr/bin/env python3
"""Perform BIC for best cluster GMM."""
import numpy as np
expectation_maximization = __import__('8-EM').expectation_maximization


def BIC(X, kmin=1, kmax=None, iterations=1000, tol=1e-5, verbose=False):
    """Perform BIC for best cluster GMM."""
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None, None

    if not isinstance(kmin, int) or kmin < 1:
        return None, None, None, None

    if not isinstance(kmax, int) or kmax < 1:
        return None, None, None, None

    if kmax - kmin < 1:
        return None, None, None, None

    if not isinstance(iterations, int) or iterations < 1:
        return None, None, None, None, None

    if not isinstance(tol, float) or tol < 0:
        return None, None, None, None, None

    if not isinstance(verbose, bool):
        return None, None, None, None, None

    BIC = []
    res = []
    logl = []
    k_s = list(range(kmin, kmax+1))
    n, d = X.shape

    for k in k_s:
        pi, m, S, _, L = expectation_maximization(X, k, iterations,
                                                  tol, verbose)
        logl.append(L)
        res.append((pi, m, S))
        p = k*d + k*(d*(d+1)/2) + k-1
        bic = p * np.log(n) - 2 * L
        BIC.append(bic)
    best = np.argmin(BIC)
    return k_s[best], res[best], np.array(logl), np.array(BIC)
