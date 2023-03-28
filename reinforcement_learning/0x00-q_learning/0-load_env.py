#!/usr/bin/env python3
"""Load OpenAI gym environment."""
import gym

def load_frozen_lake(desc=None, map_name=None, is_slippery=None):
    """
    Load the frozen lake gym environment
    for Q learning
    """
    env = gym.make('FrozenLake-v1', desc=desc, map_name=map_name,
                   is_slippery=is_slippery, render_mode='ansi')
    return env
