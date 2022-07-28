#!/usr/bin/env python3
"""Cat playing with numpy."""
import numpy as np


def np_cat(mat1, mat2, axis=0):
    """Concatinate arrays."""
    return np.concatenate(mat1, mat2, axis)
