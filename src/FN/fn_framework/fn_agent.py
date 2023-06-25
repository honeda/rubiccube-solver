from collections import namedtuple

import numpy as np

from src.env import Environment


Experience = namedtuple("Experience",
                        ["s", "a", "r", "n_s", "d"])


class FNAgent:

    def __init__(self, epsilon, actions):
        self.epsilon = epsilon
        self.actions = actions
        self.model = None
        self.estimate_probs = False
        self.initialized = False

    def save(self, model_path):
        self.model.save(model_path)

    def load(cls, env: Environment, model_path, epsilon=0.0001):
        actions = list(range(env.actions))
        agent = cls(epsilon, actions)
        # agent.model = # load model
        agent.initialized = True
        return agent

    def initialize(self, experiences):
        raise NotImplementedError

    def estimate(self, s):
        raise NotImplementedError

    def policy(self, s):
        if (np.random.random() < self.epsilon) or not self.initialized:
            return np.random.choice(self.actions)
        else:
            estimates = self.estimate(s)
            if self.estimate_probs:
                action = np.random.choice(self.actions, p=estimates)

                return action
            else:
                return np.argmax(estimates)

    def play(self, env: Environment, episode_count=5):
        for e in range(episode_count):
            s = env.reset_to_gamestart()
            done = False
            episode_reward = 0
            while not done:
                a = self.policy(s)
                n_state, reward, done = env.step(a)
                episode_reward += reward
                s = n_state
