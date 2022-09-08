#!/usr/bin/env python3
"""L2 Regulazation."""
import numpy as np

def l2_reg_cost(cost, lambtha, weights, L, m):
    """L2 regulazation cost."""
    ws = 0.0
    for la in range(0, L):
        ws += np.linalg.norm(weights['W'+str(la+1)]) ** 2

    l2 = lambtha * ws
    reg = cost + (l2/(2*m))
    return reg

