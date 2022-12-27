#!/usr/bin/env python3
"""Find the if the matrix is absorbing."""
import numpy as np


def find_path(G, dp):
    """Find if there's a path to the absorbing state."""
    for d in dp:
        if d not in G:
            for t in dp[d]:
                if t in G:
                    G.append(d)
                    find_path(G, dp)
    return G


def absorbing(P):
    """Find the if the matrix is absorbing."""
    if type(P) is not np.ndarray:
        return None
    n, p = P.shape
    if n != p:
        return None
    absorbing_states = np.argwhere(P == 1)
    if len(absorbing_states) == 0:
        return False
    abs_states = [x[0] for x in absorbing_states]
    np.fill_diagonal(P, 0)
    paths = np.argwhere(P != 0)
    dic_paths = {x[0]: [] for x in paths}
    for dp in paths:
        dic_paths[dp[0]].append(dp[1])
    fp = find_path(abs_states, dic_paths)
    for i in range(n):
        if i not in fp:
            return False
    return True
