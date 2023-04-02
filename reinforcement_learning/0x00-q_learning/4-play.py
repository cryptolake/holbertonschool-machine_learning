#!/usr/bin/env python3
"""Play an episode."""
import numpy as np


def play(env, Q, max_steps=100):
    """
    Play an openai gym episode with a Q table.

    params:
    env: OAI environment
    Q: Q table
    max_steps: max_steps in the env

    Returns:
    Total_rewards: total rewards of the episode
    """
    state, _ = env.reset()
    done = False
    total_reward = 0
    print(env.render())
    for _ in range(max_steps):
        state, reward, done, _, _ = env.step(np.argmax(Q[state]))
        print(env.render())
        total_reward += reward
        if done:
            break
        env.close()
    return total_reward
