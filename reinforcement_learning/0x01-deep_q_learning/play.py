#!/usr/bin/env python3
"""
Play atari breakout.

Reference: https://keras.io/examples/rl/deep_q_network_breakout/
"""
from Atarirl import AtariRL


if __name__ == '__main__':
    dqn = AtariRL('Breakout-v4', policy=None)
    dqn.load()
    dqn.play()
