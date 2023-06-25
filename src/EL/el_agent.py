import json
from logging import getLogger, config

import numpy as np
import matplotlib.pyplot as plt

# from src.env.action import ACTION_CHARS


class ELAgent():

    def __init__(self, epsilon):
        self.Q = {}
        self.epsilon = epsilon
        self.reward_log = []

        # load logger
        conf_file = "config/logger.json"
        log_conf = json.load(open(conf_file))
        log_conf["handlers"]["fileHandler"]["filename"] = f"log/{self.__class__.__name__}.log"

        config.dictConfig(log_conf)
        self.logger = getLogger("simpleLogger")

    def policy_base(self, s, actions):
        if np.random.random() < self.epsilon:
            return np.random.choice(actions)
        else:
            if s in self.Q and sum(self.Q[s]) != 0:
                return np.argmax(self.Q[s])
            else:
                return np.random.choice(actions)

    def policy(self, s, prev_action, actions):

        # 初手の場合
        if prev_action is None:
            return self.policy_base(s, actions)

        # 初手でない場合
        else:
            if prev_action % 2 == 0:
                kinjite = prev_action + 1  # Like F -> F_
            else:
                kinjite = prev_action - 1  # Like F_ -> F

            a = int(kinjite)
            count = 0
            while a == kinjite:
                count += 1
                a = self.policy_base(s, actions)
            # if count > 40:
            #     self.logger.info(f"prev_action={ACTION_CHARS[prev_action]}, "
            #                      f"{s=}, {dict(zip(ACTION_CHARS, self.Q[s]))}")

            return a

    def init_log(self):
        self.reward_log = []

    def log(self, reward):
        self.reward_log.append(reward)

    def show_reward_log(self, interval=50, episode=-1):
        if episode > 0:
            rewards = self.reward_log[-interval:]
            mean = np.mean(rewards)
            std = np.std(rewards)
            print(f"At Episode {episode} average reward is {mean:.2f} (+/-{std:.3f}).")
        else:
            indices = list(range(0, len(self.reward_log), interval))
            means = []
            stds = []
            for i in indices:
                rewards = self.reward_log[i:(i + interval)]
                means.append(np.mean(rewards))
                stds.append(np.std(rewards))
            means = np.array(means)
            stds = np.array(stds)
            plt.figure()
            plt.title("Reward History")
            plt.grid()
            plt.fill_between(indices, means - stds, means + stds,
                             alpha=0.1, color="g")
            plt.plot(indices, means, "o-", color="g",
                     label="Rewards for each {} episode".format(interval))
            plt.legend(loc="best")
            plt.show()
