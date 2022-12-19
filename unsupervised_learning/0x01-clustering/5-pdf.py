#!/usr/bin/env python3
"""PDF of the gaussian distribution (multivariate)."""
import numpy as np


def pdf(X, m, S):
    """PDF of the gaussian distribution (multivariate)."""
    _, d = X.shape
    x_m = (X - m).T
    P = (1. / (np.sqrt((2*np.pi)**d * np.linalg.det(S))) *
         np.exp(-(np.linalg.solve(S, x_m).T.dot(x_m)) / 2.))
    return P.diagonal()
