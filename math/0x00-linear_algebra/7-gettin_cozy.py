#!/usr/bin/env python3
"""Cat matrices."""


def deep_copy(A):
    """Deep copy."""
    return [r.copy() for r in A]


def cat_matrices2D(mat1, mat2, axis=0):
    """Cat matrices."""
    mat = deep_copy(mat1)
    if axis == 0:
        for line in mat2:
            mat.append(line)
        return mat
    elif axis == 1:
        for i, col in enumerate(mat2):
            for ele in col:
                mat[i].append(ele)
        return mat
    else:
        return None
