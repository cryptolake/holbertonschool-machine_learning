#!/usr/bin/env python3
import numpy as np
"""Perform a markov chain."""


def markov_chain(P, s, t=1):
    """Perform a markov chain."""
    if type(P) is not np.ndarray:
        return None
    n, p = P.shape
    if n != p:
        return None
    if type(s) is not np.ndarray or s.shape != (1, n):
        return None
    res = s @ np.linalg.matrix_power(P, t)
    return res
