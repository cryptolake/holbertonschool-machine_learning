#!/usr/bin/env python3
"""Matrix transpose."""


def matrix_transpose(matrix):
    """Matrix transpose."""
    result = []
    x = True
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            if x:
                result.append([])
            result[j].append(matrix[i][j])
        x = False
    return result
