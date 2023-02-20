#!/usr/bin/env python3
"""
The Viretbi Algorithm.

The Viterbi algorithm is a dynamic programming algorithm
for obtaining the maximum a posteriori probability estimate
of the most likely sequence of hidden states—called
the Viterbi path—that results in a sequence of observed events.
"""
import numpy as np


def viterbi(Observation, Emission, Transition, Initial):
    """Get the most likely sequence of hidden states with viterbi Algorithm."""
    T = Observation.shape[0]
    N = Emission.shape[0]

    full_states = np.empty((T-1, N), dtype=int)
    mu = Initial * Emission[:, Observation[0]]

    # In here we produce two variables one is the a chain of the best paths
    # we will produce an array of T arrays each array will be of size of the
    # hiden states, each element will be an index to indecate which state
    # is the best path as we go in reverse.

    for t in range(1, T):
        pre_max = mu * Transition.T
        full_states[t-1, :] = np.argmax(pre_max, 1)
        f1 = np.max(pre_max, 1)
        mu = f1 * Emission[:, Observation[t]]

    # final state which will decide the best path to take
    state = np.argmax(mu)
    sequence_prob = mu[state]
    Path = [state]
    # we get the state with the best probability from the last mu
    # then we iterate through the states in reverse to get the past path.
    for prev_states in np.flip(full_states, 0):
        state = prev_states[state]
        Path.append(state)

    return Path[::-1], sequence_prob
