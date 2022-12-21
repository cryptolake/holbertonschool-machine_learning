#!/usr/bin/env python3
"""Perform BIC for best cluster GMM."""
import numpy as np

expectation_maximization = __import__('8-EM').expectation_maximization


def BIC(X, kmin=1, kmax=None, iterations=1000, tol=1e-5, verbose=False):
    """Perform BIC for best cluster GMM."""
    if type(X) is not np.ndarray or len(X.shape) != 2:
        return None, None, None, None

    if type(iterations) is not int or iterations <= 0:
        return None, None, None, None

    if type(tol) is not float:
        return None, None, None, None

    if type(verbose) is not bool:
        return None, None, None, None

    if type(kmin) is not int or kmin < 1:
        return None, None, None, None

    if kmax is None:
        kmax = X.shape[0]

    if type(kmax) is not int or kmax < 1:
        return None, None, None, None

    BIC = []
    res = []
    logl = []
    k_s = list(range(kmin, kmax+1))
    n, d = X.shape

    for k in k_s:
        pi, m, S, _, ll = expectation_maximization(X, k, iterations,
                                                   tol, verbose)
        logl.append(ll)
        res.append((pi, m, S))
        p = k*d + k*(d*(d+1)/2) + k-1
        bic = p * np.log(n) - 2 * ll
        BIC.append(bic)
    best = np.argmin(BIC)
    return k_s[best], res[best], np.array(logl), np.array(BIC)
