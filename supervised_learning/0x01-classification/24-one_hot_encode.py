#!/usr/bin/env python3
"""One hot encoding."""

import numpy as np


def one_hot_encode(Y, classes):
    """One hot encoding on classes."""
    if type(classes) is not int:
        return None
    if classes < 2:
        return None
    y_one_hot = np.zeros((Y.size, classes))
    y_one_hot[Y, np.arange(classes)] = 1
    return y_one_hot
