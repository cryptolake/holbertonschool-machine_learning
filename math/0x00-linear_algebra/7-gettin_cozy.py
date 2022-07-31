#!/usr/bin/env python3
"""Cat matrices."""


def shape(ele, shp):
    """Recursive function to get shape."""
    if not isinstance(ele, list):
        return shp
    shp.append(len(ele))
    return shape(ele[0], shp)


def matrix_shape(matrix):
    """Caller for the recur func."""
    return shape(matrix, [])


def deep_copy(A):
    """Deep copy."""
    return [r.copy() for r in A]


def cat_matrices2D(mat1, mat2, axis=0):
    """Cat matrices."""
    if mat1 is None or mat2 is None:
        return None
    mat = deep_copy(mat1)
    shap1, shap2 = matrix_shape(mat1), matrix_shape(mat2)
    if axis == 0:
        if shap1[1] != shap2[1]:
            return None
        for line in mat2:
            mat.append(line)
        return mat
    elif axis == 1:
        for i, col in zip(range(len(mat1)), mat2):
            if col is not None:
                for ele in col:
                    mat[i].append(ele)
        return mat
    else:
        return None
