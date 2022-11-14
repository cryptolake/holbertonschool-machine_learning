#!/usr/bin/env python3
"""Find matrix of minors."""


def cross_prod(matrix):
    """Get cross product."""
    if len(matrix) == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
    return matrix[0][0]


def minor(matrix):
    """Find minor matrix."""
    for x in matrix:
        if type(x) is not list:
            raise TypeError('matrix must be a list of lists')
    if len(matrix) == 1:
        return 1
    lm = len(matrix[0])
    for s in matrix:
        if len(s) != lm or len(s) != len(matrix):
            raise ValueError('matrix must be a square matrix')
        lm = len(s)
    matmin = []
    for x in range(len(matrix)):
        mat = matrix.copy()
        mat.pop(x)
        matmin.append([])
        for y in range(len(matrix)):
            matt = mat.copy()
            for j in range(len(matt)):
                matt[j] = mat[j].copy()
            for m in matt:
                m.pop(y)
            matmin[x].append(cross_prod(matt))
    return matmin
