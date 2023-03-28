#!/usr/bin/env python3
"""Implement epsilon greedy."""
import numpy as np

def epsilon_greedy(Q, state, epsilon):
    """
    Implement epsilon greedy.

    Q: Q table
    state: current state
    epsilon: exploration rate

    Return: next action index
    """

    actions = Q.shape[1]
    p = np.random.uniform(0, 1)
    if p < epsilon:
        return np.random.randint(0, actions)
    return np.argmax(Q[state])    
