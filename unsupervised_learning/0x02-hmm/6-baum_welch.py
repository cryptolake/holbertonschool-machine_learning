#!/usr/bin/env python3
"""
Baum Welch algorithm.

The Baum–Welch algorithm is a special case of the expectation–maximization
algorithm used to find the unknown parameters of a hidden Markov model (HMM).
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


def baum_welch(Observations, Transition, Emission, Initial, iterations=1000):
    """Perform the Baum Welch algorithm."""
    # Check this wonderful medium article for reference:
    # https://medium.com/mlearning-ai/baum-welch-algorithm-4d4514cf9dbe
    # TODO: Compare this code to the probabilistic breakdown
    # for more understanding
    M = Transition.shape[0]
    T = Observations.shape[0]
    for _ in range(iterations):
        _, alpha = forward(Observations, Emission, Transition, Initial)
        _, beta = backward(Observations, Emission, Transition, Initial)
        xi = np.empty((M, M, T-1))
        for t in range(T - 1):
            denominator = (alpha[:, t] @ Transition *
                           Emission[:, Observations[t+1]].T) @ beta[:, t+1]
            for i in range(M):
                numerator = (alpha[i, t] * Transition[i, :] *
                             Emission[:, Observations[t+1]].T * beta[:, t+1].T)
                xi[i, :, t] = numerator / denominator
        gamma = np.sum(xi, 1)
        Transition = np.sum(xi, 2) / np.reshape(np.sum(gamma, 1), (-1, 1))
        gamma = np.hstack((gamma, np.reshape(np.sum(xi[:, :, T - 2], 0),
                                             (-1, 1))))

        K = Emission.shape[1]
        denominator = np.sum(gamma, 1)
        for k in range(K):
            Emission[:, k] = np.sum(gamma[:, Observations == k], 1)
        Emission = Emission / np.reshape(denominator, (-1, 1))
    return Transition, Emission
