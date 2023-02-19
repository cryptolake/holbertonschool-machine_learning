#!/usr/bin/env python3
"""
Forward algorithm.

The forward algorithm, in the context of a hidden Markov model (HMM),
is used to calculate a 'belief state': the probability of a state
at a certain time, given the history of evidence.
The process is also known as filtering.
"""
import numpy as np


def forward(Observation, Emission, Transition, Initial):
    """Perform the forward algorithm."""
    T = Observation.shape[0]
    N = Emission.shape[0]
    F = np.ndarray(shape=(N, T))

    prev = Initial
    for t in range(T):
        F[:, t] = Emission[:, Observation[t]] * np.reshape(prev, (N))
        prev = F[:, t] @ Transition
    P = np.sum(prev)
    return P, F
