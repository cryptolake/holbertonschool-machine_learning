#!/usr/bin/env python3
"""Matrix addition."""


def shape(ele, shp):
    """Recursive function to get shape."""
    if not isinstance(ele, list):
        return shp
    shp.append(len(ele))
    return shape(ele[0], shp)


def add_matrices2D(mat1, mat2):
    """Matrix addition."""
    if len(mat1) == 0 or len(mat2) == 0:
        return None
    if shape(mat1, []) != shape(mat2, []):
        return None
    mat = []
    for i in range(len(mat1)):
        mat.append([])
        for j in range(len(mat1[0])):
            mat[i].append(mat1[i][j]+mat2[i][j])
    return mat
