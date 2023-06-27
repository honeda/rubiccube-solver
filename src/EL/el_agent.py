import datetime
import pickle
import json
from collections import defaultdict
from pathlib import Path
from logging import getLogger, config

import numpy as np
import matplotlib.pyplot as plt

from src.env.cube import Cube
from src.env.action import ACTION_NUMS
from src.utils.cube_util import encode_state, decode_state, get_color_swap_states


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

    def checkpoint(self, states, Q_filename, Q_filedir):
        """squeeze -> deploy -> save
        """
        self.squeeze_q()
        self.check_done_state_values_error()
        self.deploy_q_to_swapped_state(states)
        self.save_q_file(Q_filename, Q_filedir)

    def squeeze_q(self):
        """Qのvalueの合計値が0のkeyを削除して容量削減.
        """
        key_q = np.array(list(self.Q.keys()))
        value_q = np.array(list(self.Q.values()))
        sum_q = np.sum(value_q, axis=1)

        mask = (sum_q != 0)

        dic = defaultdict(lambda: [0] * len(ACTION_NUMS))
        for k, v in zip(key_q[mask], value_q[mask]):
            dic[k] = v

        self.Q = dic

    def save_q_file(self, Q_filename, Q_filedir):
        # Save Q
        dt = datetime.datetime.now()
        filename = ("Q_{}.pkl".format(dt.strftime("%Y%m%d%H%M"))
                    if Q_filename is None else Q_filename)
        with open(Path(Q_filedir, filename), "wb") as f:
            pickle.dump(dict(self.Q), f)
        self.logger.info(f"Save Q file. {len(self.Q)=:,}")

    def load_q_file(self, file_path):
        """
        Args:
            file_path (str): file path

        Returns:
            Q (defaultdict)
            N (defaultdict)
        """
        Q = defaultdict(lambda: [0] * len(ACTION_NUMS))

        Q_ = pickle.load(open(file_path, "rb"))
        # dict -> defaultdict
        for k, v in Q_.items():
            Q[k] = v

        self.logger.info(f"Load Q file. {len(Q)=:,}")

        return Q

    def deploy_q_to_swapped_state(self, states):
        """色とアクションをスワップする技術を利用して、すでにQに保存されている局面の価値を
        色をスワップした別の局面にも反映する.
        局面の複雑具合によるが、1つの局面から最大25局面増やせる.
        """
        self.logger.info(f"Deploy Q values to swapped states. {len(self.Q)=}")
        count = 0
        for state in np.unique(states):
            if state in self.Q:
                values = self.Q[state]
                state = decode_state(state)

                if sum(values) != 0:
                    cube = Cube()
                    cube.state = state
                    swapped_cubes, translated_action_dics = get_color_swap_states(cube)
                    for sc, ta in zip(swapped_cubes, translated_action_dics):
                        swapped_state = encode_state(sc)
                        if swapped_state in self.Q:
                            continue
                        else:
                            # アクションの入れ替え
                            new_values = list(np.array(values)[ta])

                            self.Q[swapped_state] = new_values
                            count += 1
                else:
                    print("yeah")

        self.logger.info(f"Complete. Deployed {count:,} states.")
        try:
            print(f"LAST {swapped_state=}")
        except Exception:
            pass

    def check_done_state_values_error(self):
        """完成状態のstateの価値がすべて0であることを確認する."""
        solved_state = encode_state(Cube())
        if max(self.Q[solved_state]) > 0:
            # save Q
            dt = datetime.datetime.now()
            filename = "err_Q_{}.pkl".format(dt.strftime("%Y%m%d%H%M"))
            dir_ = "data/EL/"
            self.save_q_file(filename, dir_)

            msg = (f"Q-value of solved state is not '0'. {self.Q[solved_state]=}."
                   f" Save Q file -> {Path(dir_, filename)}.")
            self.logger.critical(msg)

            raise Exception
