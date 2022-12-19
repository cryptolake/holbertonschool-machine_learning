#!/usr/bin/env python3
"""PDF of the gaussian distribution (multivariate)."""
import numpy as np


def pdf(X, m, S):
    """PDF of the gaussian distribution (multivariate)."""
    _, d = X.shape
    x_m = X - m
    det = np.linalg.det(S)
    inv = np.linalg.inv(S)
    num = np.exp(-0.5 * np.sum(x_m * np.matmul(inv, x_m.T).T, axis=1))
    denum = np.sqrt(((2*np.pi)**d)*det)
    return num/denum
