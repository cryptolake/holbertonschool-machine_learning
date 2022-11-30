#!/usr/bin/env python3
"""t-SNE the new dimensionality reduction method."""
import numpy as np


def P_init(X, perplexity):
    """Initialize P in t-SNE."""
    # Math? See https://stackoverflow.com/questions/37009647
    sum_X = np.sum(np.square(X), 1)
    D = np.add(np.add(-2 * np.dot(X, X.T), sum_X).T, sum_X)
    np.fill_diagonal(D, 0)
    P = np.zeros((X.shape[0], X.shape[0]))
    betas = np.ones((X.shape[0], 1))
    H = np.log2(perplexity)
    return D, P, betas, H
