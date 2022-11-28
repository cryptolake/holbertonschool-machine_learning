#!/usr/bin/env python3
"""Simple principle componenet analysis."""
import numpy as np


def pca(X, var=0.95):
    """Perform pca while keeping features that cover at least var variance."""
    _, s, v = np.linalg.svd(X)
    cum_var = np.cumsum(s) / np.sum(s)
    arg = np.argwhere(cum_var >= var)[0, 0]
    return v[:arg+1].T
