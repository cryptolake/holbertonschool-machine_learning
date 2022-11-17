#!/usr/bin/env python3
"""Get Correlation matrix."""
import numpy as np


def correlation(C):
    """Get correlation matrix."""
    if type(C) is not np.ndarray:
        raise TypeError("C must be a numpy.ndarray")
    if len(C.shape) != 2 or C.shape[0] != C.shape[1]:
        raise ValueError("C must be a 2D square matrix")
    D = np.sqrt(np.diag(C))
    dinv = np.linalg.inv(np.diag(D))
    corr = dinv @ C @ dinv
    return corr.astype(float)
