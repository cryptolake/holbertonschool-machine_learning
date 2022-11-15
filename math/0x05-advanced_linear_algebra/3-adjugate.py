#!/usr/bin/env python3
"""Find adjugate of matrix."""


def determinant(matrix):
    """Find Determinant."""
    for x in matrix:
        if type(x) is not list:
            raise TypeError('matrix must be a list of lists')
    if len(matrix) == 1:
        if len(matrix[0]) == 0:
            return 1
        else:
            return matrix[0][0]
    lm = len(matrix[0])
    for s in matrix:
        if len(s) != lm or len(s) != len(matrix):
            raise ValueError('matrix must be a square matrix')
        lm = len(s)
    if len(matrix) == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
    else:
        b = 0
        c = 1
        mat = matrix.copy()
        mat.pop(0)
        for i in range(len(matrix)):
            matt = mat.copy()
            for j in range(len(matt)):
                matt[j] = mat[j].copy()
            for x in matt:
                x.pop(i)
            b += c * matrix[0][i] * determinant(matt)
            c *= -1
        return b


def cofactor(matrix):
    """Find cofactor matrix."""
    for x in matrix:
        if type(x) is not list:
            raise TypeError('matrix must be a list of lists')
    if len(matrix) == 0:
        raise TypeError('matrix must be a non-empty square matrix')
    if len(matrix) == 1 and len(matrix[0]) == 1:
        return [[1]]
    lm = len(matrix[0])
    for s in matrix:
        if len(s) != lm or len(s) != len(matrix):
            raise ValueError('matrix must be a non-empty square matrix')
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
            matmin[x].append(((-1)**(x+y+2))*determinant(matt))
    return matmin


def adjugate(matrix):
    """Get adjugate matrix."""
    am = cofactor(matrix)
    pairs = []
    for x in range(len(am)):
        for y in range(len(am)):
            if x != y and (x, y) not in pairs and (y, x) not in pairs:
                s = am[x][y]
                am[x][y] = am[y][x]
                am[y][x] = s
                pairs.append((x, y))
                pairs.append((y, x))
    return am
