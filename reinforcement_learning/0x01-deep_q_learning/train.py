#!/usr/bin/env python3
"""
Training of a DQN agent to play atari breakout.

Reference: https://keras.io/examples/rl/deep_q_network_breakout/
"""
from Atarirl import AtariRL, EpsilonGreedy


if __name__ == '__main__':
    policy = EpsilonGreedy(epsilon_greedy_frames=1000000.0)
    dqn = AtariRL('Breakout-v4', policy=policy.explore,
                  max_memory_length=80000)
    dqn.train()
    dqn.save()
