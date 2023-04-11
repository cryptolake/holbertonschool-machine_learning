"""
DQN agent to train/play atari breakout.

Reference: https://keras.io/examples/rl/deep_q_network_breakout/
"""
import gym
from gym.wrappers import AtariPreprocessing, FrameStack
import gc
gc.enable()
from tensorflow.keras import layers, models, optimizers, losses
import tensorflow.keras as K
import numpy as np
import sys
import tensorflow as tf


class EpsilonGreedy:
    """
    Epsilon greedy for Deep-QN.
    """

    def __init__(self, epsilon=1.0, min_epsilon=0.1, max_epsilon=1.0,
                 epsilon_greedy_frames=1000000.0, epsilon_random_frames=50000):
        """Initialize Epsilon Greedy."""
        self.epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.epsilon_interval = max_epsilon - min_epsilon
        self.epsilon_greedy_frames = epsilon_greedy_frames
        self.epsilon_random_frames = epsilon_random_frames

    def __update(self):
        """Get the next epsilon value."""
        epsilon = self.epsilon - self.epsilon_interval / self.epsilon_greedy_frames
        self.epsilon = max(epsilon, self.min_epsilon)
    def explore(self, frame_count):
        """Decide whether to explore or not."""
        ex = frame_count < self.epsilon_random_frames or self.epsilon > np.random.uniform(0, 1)
        self.__update()
        return ex


class AtariRL:
    """
    This class represents the environment setup and all
    the preprocessing with a deep QN for training at the atari
    game Breakout in the gym library.
    """

    def __init__(self, env, policy, window_length=4,
                 input_shape=(84, 84), batch_size=32, max_memory_length=10000):
        """Initialize breakout env."""
        self.window_length = window_length
        self.input_shape = input_shape
        self.env_name = env
        self.env = FrameStack(
            AtariPreprocessing(gym.make(env, frameskip=1, full_action_space=False)),
            window_length)
        self.__experience_info = ['states', 'actions',
                                  'rewards', 'next_states', 'done']
        self.memory = {x : [] for x in self.__experience_info}
        self.na = self.env.action_space.n
        self.policy = policy
        # The first model makes the predictions for Q-values which are used to
        # make a action.
        self.model = self.__create_model()
        # Build a target model for the prediction of future rewards.
        # The weights of a target model get updated every 10000 steps thus when the
        # loss between the Q-values is calculated the target Q-value is stable.
        self.target_model = self.__create_model()
        self.batch_size = batch_size
        self.max_memory_length = max_memory_length
        self.frame_count = 0


    def __create_model(self):
        """
        Create keras DQN model.
        We use the same model that was described by Mnih et al. (2015).
        """
        inp = layers.Input(shape=(self.window_length,)+self.input_shape)
        x1 = layers.Permute((2, 3, 1))(inp)
        x2 = layers.Conv2D(32, 8, strides=4, activation='relu')(x1)
        x3 = layers.Conv2D(64, 4, strides=2, activation='relu')(x2)
        x4 = layers.Conv2D(64, 3, strides=1, activation='relu')(x3)
        x5 = layers.Flatten()(x4)
        x6 = layers.Dense(512, activation='relu')(x5)
        y = layers.Dense(self.na, activation='linear')(x6)
        return models.Model(inputs=inp, outputs=y)

    def __add_to_memory(self, dq_tuple):
        if len(self.memory['states']) > self.max_memory_length:
            for exp in self.__experience_info:
                del self.memory[exp][:1]
        for i, exp in enumerate(self.__experience_info):
            self.memory[exp].append(dq_tuple[i])

    def __random_memory_sample(self):
        indices = np.random.choice(range(len(self.memory['states'])),
                                   size=self.batch_size)
        sample = {}
        for exp in self.__experience_info:
            sample[exp] = tf.convert_to_tensor([self.memory[exp][i] for i in indices],
                                               dtype=tf.float32)
        return sample

    def train(self, max_steps=10000, lr=0.00025, update_after=4,
              update_target=10000, gamma=0.99, reward_threshold=40,
              episode_memory_size=100, checkpoint=1000000):
        """Training loop."""
        self.optimizer = optimizers.Adam(learning_rate=lr, clipnorm=1.0)
        self.loss = losses.Huber()
        episode_history = []
        while True:
            state, _ = self.env.reset()
            state = np.array(state, dtype=np.float32)
            done = False
            episode_reward = 0
            for _ in range(max_steps):
                self.frame_count += 1
                if self.policy(self.frame_count):
                    action = np.random.choice(self.na)
                else:
                    state_tensor = tf.convert_to_tensor(state)
                    # batch dimension
                    state_tensor = tf.expand_dims(state_tensor, 0)
                    action_probs = self.model(state_tensor, training=False)
                    # take the action with most probability
                    action = tf.argmax(action_probs, 1).numpy()[0]

                state_next, reward, done, _, _ = self.env.step(action)
                
                state_next = np.array(state_next, dtype=np.float32)
                self.__add_to_memory((state, action, reward, state_next, done))

                episode_reward += reward

                if self.frame_count % update_after == 0 and len(self.memory['states']) > self.batch_size:

                    sample = self.__random_memory_sample()

                    # Build the updated Q-values for the sampled future states
                    # Use the target model for stability
                    future_rewards = self.target_model(sample['next_states'], training=False)

                    # Q value = reward + discount factor * expected future reward
                    updated_q_values = sample['rewards'] + gamma * tf.reduce_max(future_rewards, axis=1)

                    # If final frame set the last value to -1
                    updated_q_values = updated_q_values * (1 - sample['done']) - sample['done']

                    sample_actions = tf.cast(sample['actions'], tf.int32)
                    masks = tf.one_hot(sample_actions, self.na)

                    with tf.GradientTape() as tape:
                        q_values = self.model(sample['states'])
                        q_action = tf.reduce_max(tf.multiply(q_values, masks), axis=1)
                        loss = self.loss(updated_q_values, q_action)

                    sys.stdout.flush()

                    grads = tape.gradient(loss, self.model.trainable_variables)
                    self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

                    if self.frame_count % update_target == 0:
                        self.target_model.set_weights(self.model.get_weights())

                    if checkpoint != None:
                        if self.frame_count % checkpoint:
                            self.save(f"{self.frame_count}-checkpoint.h5")
                
                state = state_next

                if done:
                    print(f"{self.frame_count} frames done")
                    break

            episode_history.append(episode_reward)
            if len(episode_history) > episode_memory_size:
                del episode_history[:1]
            running_reward = np.mean(episode_history)

            if running_reward > reward_threshold:
                break
            gc.collect()
            K.backend.clear_session()

    def play(self, nb_episodes=10, max_steps=1000):
        """Let the DQN play the game."""

        self.env = FrameStack(
            AtariPreprocessing(gym.make(self.env_name, render_mode='human',
                                        frameskip=1, full_action_space=False)),
            self.window_length)
        for _ in range(nb_episodes):
            done = False
            state, _ = self.env.reset()
            i = 0
            while not done and i < max_steps:
                state = tf.expand_dims(state, 0)
                pred = self.model.predict(state)
                action = tf.argmax(pred, 1).numpy()[0]
                state_next, _, done, _, _ = self.env.step(action)
                state = state_next
                i+=1


    def save(self, weights="Breakout_weights.h5"):
        """Save DQN model weights."""
        self.model.save_weights(weights)

    def load(self, weights="Breakout_weights.h5"):
        """Load DQN from weights file."""
        self.model.load_weights(weights)
