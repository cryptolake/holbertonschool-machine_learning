#!/usr/bin/env python3
"""Q-learning."""
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


def train(env, Q, episodes=5000, max_steps=100, alpha=0.1, gamma=0.99,
          epsilon=1, min_epsilon=0.1, epsilon_decay=0.55):
    """
    Perform Q-learning.

    params:
    env: OpenAI gym env
    Q: Q-table
    episodes: number of episodes, each episode has it's own
    total reward, and we reset then env at each ep
    max_steps: max steps in each episode
    alpha: the learning rate
    gamma: the discount rate
    epsilon: the initial threshold for epsilon greedy
    min_epsilon: the minimum value that epsilon should decay to
    epsilon_decay: the decay rate for updating epsilon between episodes

    Return:
    Q: the update Q-table
    total_rewards: rewards per episode
    """
    total_rewards = []
    done = False
    for _ in range(episodes):
        state, _ = env.reset()
        ep_rewards = 0
        for _ in range(max_steps):
            action = epsilon_greedy(Q, state, epsilon)
            next_state, reward, done, _, _ = env.step(action)
            if done and reward == 0:
                reward = -1
            ep_rewards += reward

            old_value = Q[state, action]
            next_max = np.max(Q[next_state])

            new_value = (1 - alpha) * old_value + alpha * (reward +
                                                           gamma * next_max)

            Q[state, action] = new_value
            
            state = next_state
            if done:
                break
        epsilon = max(epsilon * epsilon_decay, min_epsilon)
        total_rewards.append(ep_rewards)
    return Q, total_rewards
