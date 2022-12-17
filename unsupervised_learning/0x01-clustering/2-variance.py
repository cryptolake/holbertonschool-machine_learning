#!/usr/bin/env python3
"""Intra cluster variance."""
import numpy as np


def variance(X, C):
    """Get Intra cluster variance.Intra cluster variance."""
    try:
        diff = X[..., np.newaxis] - C.T
        distances = np.sqrt(np.sum(diff**2, axis=1))
        return np.sum(np.min(distances, axis=1)**2)
    except Exception:
        return None
