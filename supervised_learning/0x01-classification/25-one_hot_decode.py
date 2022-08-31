#!/usr/bin/env python3
"""One hot deconding."""

import numpy as np


def one_hot_decode(one_hot):
    """One hot deconding on encoded array."""
    if type(one_hot) is not np.ndarray:
        return None
    classes = np.where(one_hot == 1)
    decoded = np.zeros((one_hot.shape[1],))
    decoded[classes[1]] = classes[0]
    return decoded.astype(int)
