#!/usr/bin/env python3
"""Get Correlation matrix."""
import numpy as np


def correlation(C):
    """Get correlation matrix."""
    if type(C) is not np.ndarray:
        raise TypeError
    if len(C.shape) != 2:
        raise ValueError
    D = np.sqrt(np.diag(C))
    dinv = np.linalg.inv(np.diag(D))
    corr = dinv @ C @ dinv
    return corr.astype(float)
