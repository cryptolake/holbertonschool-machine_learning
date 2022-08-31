#!/usr/bin/env python3
"""One hot decoding."""

import numpy as np


def one_hot_decode(one_hot):
    """One hot deconding on encoded array."""
    if type(one_hot) is not np.ndarray:
        return None
    if len(one_hot) < 2:
        return None
    return np.where(one_hot.T == 1)[1]
