#!/usr/bin/env python3
"""Find definiteness of matrix."""
import numpy as np


def definiteness(matrix):
    """Find definiteness of matrix."""
    if type(matrix) is not np.ndarray:
        raise TypeError("matrix must be a numpy.ndarray")
    if len(matrix) == 0:
        return None
    nl = matrix.shape[0]
    for s in matrix.shape[1:]:
        if s != nl or s != len(matrix.shape):
            return None
        nl = s

    if not np.array_equal(matrix, matrix.T):
        return None

    w, _ = np.linalg.eig(matrix)
    p = False
    n = False
    z = False
    for v in w:
        if v < 0:
            n = True
        elif v > 0:
            p = True
        else:
            z = True
    if p is True and n is False and z is False:
        return "Positive definite"
    elif p is True and z is True and n is False:
        return "Positive semi-definite"
    elif n is True and p is False and z is False:
        return "Negative definite"
    elif n is True and z is True and p is False:
        return "Negative semi-definite"
    if p is True and n is True:
        return "Indefinite"
