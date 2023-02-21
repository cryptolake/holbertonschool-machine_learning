#!/usr/bin/env python3
"""
Backward algorithm.

The backward algorithm, in the context of a hidden Markov model (HMM),
is used to calculate the probability of a state to generate
the next observations.
"""
import numpy as np


def backward(Observation, Emission, Transition, Initial):
    """Backward algorithm."""
    T = Observation.shape[0]
    N = Emission.shape[0]
    B = np.ndarray(shape=(N, T))

    B[:, -1] = 1.
    for t in range(T-2, -1, -1):
        obs = Emission[:, Observation[t+1]]
        for i in range(N):
            B[i, t] = np.sum(B[:, t+1] * Transition[i, :] * obs)
    # TODO: I need to know why Initial[:, 0] here
    P = np.sum(Initial[:, 0] * Emission[:, Observation[0]] * B[:, 0])
    return P, B
