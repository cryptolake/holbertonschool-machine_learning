#!/usr/bin/env python3
"""concat arrays."""


def cat_arrays(arr1, arr2):
    """Concat arrays."""
    arr = arr1.copy()
    for e in arr2:
        arr.append(e)
    return arr
