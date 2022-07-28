#!/usr/bin/env python3
"""Matrix Multiplication."""


def shape(ele, shp):
    """Recursive function to get shape."""
    if not isinstance(ele, list):
        return shp
    shp.append(len(ele))
    return shape(ele[0], shp)


def column(mat, x):
    """Get column x of matrix."""
    col = []
    for i in mat:
        col.append(i[x])
    return col


def dot_product(line, col):
    """Get dot product."""
    res = 0
    for i, j in zip(line, col):
        res += i * j
    return res


def mat_mul(mat1, mat2):
    """Matrix Multiplication."""
    mat = []
    sh1 = shape(mat1, [])
    sh2 = shape(mat2, [])
    if sh1[1] != sh2[0]:
        return None
    for i in range(len(mat1)):
        mat.append([])
        for j in range(sh2[1]):
            mat[i].append(dot_product(mat1[i], column(mat2, j)))
    return mat
