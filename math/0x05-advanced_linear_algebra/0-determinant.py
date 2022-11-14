#!/usr/bin/env python3
import numpy as np


def find_minor(index, matrix):
    """Find minor for matrix."""
    mins = []
    for i in range(len(index)):
        a = list(range(matrix.shape[i]))
        a.remove(index[i])
        mins.append(a)
    return matrix[np.ix_(*mins)]


def determinant(matrix):
    """Find Determinant."""
    matrix = np.array(matrix)
    if len(matrix.shape) == 1:
        raise TypeError('matrix must be a list of lists')
    if matrix.shape == (1, 0):
        return 1
    if matrix.shape == (1, 1):
        return matrix[0, 0]
    nl = matrix.shape[0]
    for s in matrix.shape[1:]:
        if s != nl:
            raise ValueError('matrix must be a square matrix')
    if matrix.shape == (2, 2):
        return matrix[0, 0] * matrix[1, 1] - matrix[0, 1] * matrix[1, 0]
    else:
        b = 0
        c = 1
        for i in range(matrix.shape[1]):
            b += c * matrix[0, i] * determinant(find_minor([0, i], matrix))
            c *= -1
        return b
