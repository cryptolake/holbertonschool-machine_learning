#!/usr/bin/env python3
"""Find the stationary matrix."""
import numpy as np


def regular(P):
    """Find the stationary matrix."""
    if type(P) is not np.ndarray:
        return None
    n, p = P.shape
    if n != p:
        return None
    if np.array_equal(P, np.eye(n)):
        return None
    w, v = np.linalg.eig(P.T)
    res = (v[:, np.isclose(w, 1)] / v[:, np.isclose(w, 1)].sum()).real
    res = res.reshape(res.shape[::-1])
    if len(res) == 0:
        return None
    if np.any(res < 0):
        return None
    return res
