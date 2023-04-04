#!/usr/bin/env python3
"""
Play atari breakout.

Reference: https://keras.io/examples/rl/deep_q_network_breakout/
"""
from Atarirl import AtariRL, EpsilonGreedy


if __name__ == '__main__':
    policy = EpsilonGreedy()
    dqn = AtariRL('Breakout-v4', policy=policy.explore)
    dqn.load()
    dqn.play()
