import datetime
import math
import pickle
from pathlib import Path
from collections import defaultdict

from src.EL.el_agent import ELAgent
from src.env import Environment, ACTIONS

import numpy as np


class MonteCarloAgent(ELAgent):

    def __init__(self, epsilon=0.1):
        super().__init__(epsilon)

    def learn(self, env, QN_file=None, n_theme=50, theme_steps=3, theme_actions=None,
              n_episode=1000, gamma=0.9, report_interval=100, Q_filedir="data/", Q_filename=None):
        """
        Args:
            env (Environment):
            QN_file (str, optional): Q file path. Defaults to None.
            n_theme (int, optional): Num of theme. Defaults to 50.
            theme_steps (int, optional): Num of step each theme.
                Defaults to 3.
            theme_actions (list, optional): Use when you want to give
                a specific theme. ex) [["F", "B"], ["D", "F_", "B"].
                `n_theme` and `theme_steps` are ignored when this argument
                is not None. Defaults to None.
            n_episode (int, optional): Num of episode per theme.
                Defaults to 1000.
            gamma (float, optional): update weight for Q. Defaults to 0.9.
            report_interval (int, optional): Defaults to 100.
            Q_filedir (str, optional): Defaults to "data/".
            Q_filename (_type_, optional): Defaults to None.
        """
        self.init_log()
        action_nums = list(range(len(ACTIONS)))

        self.Q = defaultdict(lambda: [0] * len(action_nums))
        N = defaultdict(lambda: [0] * len(action_nums))
        if QN_file:
            Q, N_ = pickle.load(open(QN_file, "rb"))
            # dict -> defaultdict
            for k, v in Q.items():
                self.Q[k] = v
            for k, v in N_.items():
                N[k] = v
            print(f"{len(self.Q)=}")

        n_unscrable_step = 25

        if theme_actions is None:
            theme_actions = [np.random.choice(action_nums, size=theme_steps)
                             for _ in range(n_theme)]
            # "F"の後に"F_"のように戻す動作は入れない.
            theme_actions = [self.replace_wasted_work(i) for i in theme_actions]

        for scramble_actions in theme_actions:
            # Scramble
            print("==================================")
            print(f"Theme scene: {scramble_actions}")
            env.reset_to_origin()
            env.set_game_start_position(scramble_actions)

            for e in range(n_episode):

                env.reset()
                s = env.states
                done = False
                # Play until the end of episode.
                experience = []
                # while not done:
                for _ in range(n_unscrable_step):
                    a = self.policy(s, action_nums)
                    n_state, reward, done = env.step(a)
                    experience.append({"state": s, "action": a, "reward": reward})
                    s = n_state
                    if done:
                        break

                self.log(reward)

                # Evaluate each state, action.
                for i, x in enumerate(experience):
                    s, a = x["state"], x["action"]

                    # Calculate discounted future reward of s.
                    G, t = 0, 0
                    for j in range(i, len(experience)):
                        G += math.pow(gamma, t) * experience[j]["reward"]
                        t += 1

                    N[s][a] += 1  # count of s, a pair
                    alpha = 1 / N[s][a]
                    self.Q[s][a] += alpha * (G - self.Q[s][a])

                if e != 0 and e % report_interval == 0:
                    self.show_reward_log(episode=e)

        # Save Q & N
        dt = datetime.datetime.now()
        filename = ("QN_{}.pkl".format(dt.strftime("%Y%m%d%H%M"))
                    if Q_filename is None else Q_filename)
        with open(Path(Q_filedir, filename), "wb") as f:
            pickle.dump([dict(self.Q), dict(N)], f)
            print(f"{len(self.Q)=}, {len(dict(self.Q))=}")

    def replace_wasted_work(self, actions: np.ndarray):
        """
        Args:
            actions (np.ndarray):
        Return:
            np.ndarray
        """
        a = actions.copy()
        idx1, idx2 = [0], [0]
        while not ((len(idx1) == 0) and (len(idx2) == 0)):
            diff = np.diff(a, prepend=a[0])
            # replace like "F F_" -> "F F"
            idx1 = np.argwhere((diff == 1) & (a % 2 == 1)).ravel()
            for i in idx1:
                a[i] = a[i] - 1
            # replace like "F_ F" -> "F_ F_"
            idx2 = np.argwhere((diff == -1) & (a % 2 == 0)).ravel()
            for i in idx2:
                a[i] = a[i] + 1

        return a


def _get_newest_qn_file(dir="data/"):
    files = [i for i in Path(dir).iterdir()
             if i.name.startswith("QN") and i.name.endswith(".pkl")]
    if len(files) == 0:
        return None
    else:
        dts = [datetime.datetime.strptime(i.name, "QN_%Y%m%d%H%M.pkl")
               for i in files]
        idx = dts.index(max(dts))

        return str(files[idx])


def train():
    QN_file = _get_newest_qn_file()
    print(f"{QN_file=}")
    agent = MonteCarloAgent(epsilon=0.1)
    env = Environment()
    agent.learn(env, n_theme=500, theme_steps=6, n_episode=2000,
                QN_file=QN_file, report_interval=100)
    agent.show_reward_log()


if __name__ == "__main__":
    train()
