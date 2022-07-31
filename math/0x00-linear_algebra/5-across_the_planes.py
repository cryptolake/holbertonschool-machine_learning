#!/usr/bin/env python3
"""Matrix addition."""


def shape(ele, shp):
    """Recursive function to get shape."""
    if not isinstance(ele, list):
        return shp
    shp.append(len(ele))
    return shape(ele[0], shp)


def matrix_shape(matrix):
    """Caller for the recur func."""
    if len(matrix) == 0:
        return [0]
    return shape(matrix, [])


def add_matrices2D(mat1, mat2):
    """Matrix addition."""
    if len(mat1) != len(mat2):
        return None
    if len(mat1[0]) != len(mat2[0]):
        return None
    if matrix_shape(mat1) != matrix_shape(mat2):
        return None
    mat = []
    for i in range(len(mat1)):
        mat.append([])
        for j in range(len(mat1[0])):
            mat[i].append(mat1[i][j]+mat2[i][j])
    return mat
