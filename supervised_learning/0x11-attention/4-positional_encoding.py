#!/usr/bin/env python3
"""Positional encoding."""
import numpy as np


def positional_encoding(max_seq_len, dm):
    """Positional encoding."""
    P = np.zeros((max_seq_len, dm))
    n = 10000
    for k in range(max_seq_len):
        for i in range(dm//2):
            P[k, 2*i] = np.sin(k/(n**((2*i)/dm)))
            P[k, 2*i+1] = np.cos(k/(n**((2*i)/dm)))
    return P
