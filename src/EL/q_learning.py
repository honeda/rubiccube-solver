import datetime
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np

from src.EL.el_agent import ELAgent, ACTION_NUMS
from src.env.action import int2str_actions, replace_wasted_work


def get_newest_q_file(dir_="data/EL/Q_learning"):
    files = [i for i in Path(dir_).iterdir()
             if i.name.startswith("Q") and i.name.endswith(".pkl")]
    if len(files) == 0:
        return None
    else:
        dts = [datetime.datetime.strptime(i.name, "Q_%Y%m%d%H%M.pkl")
               for i in files]
        idx = dts.index(max(dts))

        return str(files[idx])


class QLearningAgent(ELAgent):

    def __init__(self, epsilon=0.1):
        super().__init__(epsilon)

    def learn(self, env, Q_file=None, n_theme=50, n_theme_step=3, n_unscramble_step=20,
              n_episode=1000, theme_actions=None, gamma=0.9, learning_rate=0.1,
              report_interval=100, Q_filedir="data/EL/Q_learning/", Q_filename=None):
        """
        Args:
            env (Environment):
            Q_file (str, optional): Q file path. Defaults to None.
            n_theme (int, optional): Num of theme. Defaults to 50.
            n_theme_step (int, optional): Num of step each theme.
                Defaults to 3.
            n_unscramble_step (int, optional): Num of unscramble step.
            n_episode (int, optional): Num of episode per theme.
                Defaults to 1000.
            theme_actions (list, optional): Use when you want to give
                a specific theme. ex) [["F", "B"], ["D", "F_", "B"].
                `n_theme` and `theme_steps` are ignored when this argument
                is not None. Defaults to None.
            gamma (float, optional): update weight for Q. Must be 0.0 ~ 1.0.
                Defaults to 0.9.
            learning_rate (float, optional): learning rate. Defaults to 0.1.
            report_interval (int, optional): Defaults to 100.
            Q_filedir (str, optional): Defaults to "data/".
            Q_filename (_type_, optional): Defaults to None.
        """
        # Prepare
        self.init_log()

        if Q_file:
            self.Q = self.load_q_file(Q_file)
        else:
            self.Q = defaultdict(lambda: [0] * len(ACTION_NUMS))

        if theme_actions is None:
            theme_actions = np.random.choice(ACTION_NUMS, size=(n_theme, n_theme_step))
            # "F"の後に"F_"のように戻す動作は入れない.
            theme_actions = [replace_wasted_work(i) for i in theme_actions]

        # Learning
        for i, scramble_actions in enumerate(theme_actions, 1):
            # Scramble
            print("==============================================================")
            print(f"No.{i:0>4} Theme scene: {int2str_actions(scramble_actions)}")
            # self.save_theme_fig(scramble_actions, i)

            env.set_game_start_position(scramble_actions)

            done_th = n_episode / 5  # この回数完成させないと次のthemeにいかない
            e_max = n_episode * 5  # done_thに達していなくてもこのエピソード数で次へ
            n_done = 0
            e = 0
            # for e in range(n_episode):
            while (n_done < done_th) or (e < n_episode):
                e += 1

                env.reset_to_gamestart()
                s = env.states
                done = False
                for _ in range(n_unscramble_step):
                    a = self.policy(s, ACTION_NUMS)
                    n_state, reward, done = env.step(a)

                    # monte-carloと同じく手数ペナルティはgammaによって適用される.
                    gain = reward + gamma * max(self.Q[n_state])  # Q[n_state]は移行先
                    estimated = self.Q[s][a]
                    self.Q[s][a] += learning_rate * (gain - estimated)
                    s = n_state

                    if done:
                        n_done += 1
                        break

                if e == e_max:
                    break

                self.log(reward)

                if e != 0 and e % report_interval == 0:
                    self.show_reward_log(episode=e)

        self.Q = self.squeeze_q(self.Q)
        self.save_q_file(self.Q, Q_filename, Q_filedir)

    def squeeze_q(self, Q):
        """Qのvalueの合計値が0のkeyを削除して容量削減.

        Return:
            dict: NOT defaultdict
        """
        key_q = np.array(list(Q.keys()))
        value_q = np.array(list(Q.values()))
        sum_q = np.sum(value_q, axis=1)

        mask = (sum_q != 0)
        q = dict(zip(key_q[mask], value_q[mask]))

        return q

    def save_q_file(self, Q, Q_filename, Q_filedir):
        # Save Q
        dt = datetime.datetime.now()
        filename = ("Q_{}.pkl".format(dt.strftime("%Y%m%d%H%M"))
                    if Q_filename is None else Q_filename)
        with open(Path(Q_filedir, filename), "wb") as f:
            pickle.dump(dict(Q), f)
            print(f"{len(Q)=}")

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
        print(f"{len(Q)=}")

        return Q
