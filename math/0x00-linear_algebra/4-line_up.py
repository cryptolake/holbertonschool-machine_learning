#!/usr/bin/env python3
"""Array addtion."""


def add_arrays(arr1, arr2):
    """Add Arrays."""
    res = []
    if len(arr1) != len(arr2):
        return None
    for i in range(len(arr1)):
        res.append(arr2[i] + arr1[i])
    return res
