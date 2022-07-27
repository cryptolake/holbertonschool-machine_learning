#!/usr/bin/env python3
"""Get shape of array."""


def shape(ele, shp):
    """Recursive function to get shape."""
    if not isinstance(ele, list):
        return shp
    shp.append(len(ele))
    return shape(ele[0], shp)


def matrix_shape(matrix):
    """Caller for the recur func."""
    return shape(matrix, [])
