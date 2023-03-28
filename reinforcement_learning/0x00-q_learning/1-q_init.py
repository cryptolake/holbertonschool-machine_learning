#!/usr/bin/env python3
"""Initialize Q-table from OA gym env."""
import numpy as np

def q_init(env):
    """Initialize Q-table from OA gym env."""
    states, actions = env.observation_space.n, env.action_space.n
    Q = np.zeros((states, actions))
    return Q    
