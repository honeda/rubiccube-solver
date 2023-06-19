import datetime
import math
import pickle
from pathlib import Path
from collections import defaultdict

import numpy as np

from src.EL.el_agent import ELAgent
from src.env.action import int2str_actions, ACTION_CHARS
from src.env.cube import Cube
from src.env.action import steps
from src.utils.cube_util import show_cube

# Remove X, Y, Z
ACTION_CHARS = ACTION_CHARS[:-6]
ACTION_NUMS = list(range(len(ACTION_CHARS)))


class MonteCarloAgent(ELAgent):

    def __init__(self, epsilon=0.1):
        super().__init__(epsilon)

    def learn(self, env, QN_file=None, n_theme=50, n_theme_steps=3, n_unscramble_step=20,
              n_episode=1000, theme_actions=None, gamma=0.9, report_interval=100,
              Q_filedir="data/", Q_filename=None):
        """
        Args:
            env (Environment):
            QN_file (str, optional): Q file path. Defaults to None.
            n_theme (int, optional): Num of theme. Defaults to 50.
            n_theme_steps (int, optional): Num of step each theme.
                Defaults to 3.
            n_unscramble_step (int, optional): Num of unscramble step.
            n_episode (int, optional): Num of episode per theme.
                Defaults to 1000.
            theme_actions (list, optional): Use when you want to give
                a specific theme. ex) [["F", "B"], ["D", "F_", "B"].
                `n_theme` and `theme_steps` are ignored when this argument
                is not None. Defaults to None.
            gamma (float, optional): update weight for Q. Defaults to 0.9.
            report_interval (int, optional): Defaults to 100.
            Q_filedir (str, optional): Defaults to "data/".
            Q_filename (_type_, optional): Defaults to None.
        """
        # Prepare
        self.init_log()

        self.Q = defaultdict(lambda: [0] * len(ACTION_NUMS))
        N = defaultdict(lambda: [0] * len(ACTION_NUMS))

        if QN_file:
            Q, N_ = pickle.load(open(QN_file, "rb"))
            # dict -> defaultdict
            for k, v in Q.items():
                self.Q[k] = v
            for k, v in N_.items():
                N[k] = v
            print(f"{len(self.Q)=}")

        if theme_actions is None:
            theme_actions = [np.random.choice(ACTION_NUMS, size=n_theme_steps)
                             for _ in range(n_theme)]
            # "F"の後に"F_"のように戻す動作は入れない.
            theme_actions = [self.replace_wasted_work(i) for i in theme_actions]

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
                # Play until the end of episode.
                experience = []
                # while not done:
                for _ in range(n_unscramble_step):
                    a = self.policy(s, ACTION_NUMS)
                    n_state, reward, done = env.step(a)
                    experience.append({"state": s, "action": a, "reward": reward})
                    s = n_state
                    if done:
                        n_done += 1
                        break

                if e == e_max:
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

        self.save_qn_file(self.Q, N, Q_filename, Q_filedir)

    def save_theme_fig(self, scramble_actions, theme_num):
        dummy_cube = Cube()
        steps(dummy_cube, scramble_actions)

        dir_ = "data/figure"
        filename = f"/{theme_num:0>4}_{'-'.join(int2str_actions(scramble_actions))}.png"
        show_cube(dummy_cube, save=dir_ + filename)

    # X, Y, Z を使わない場合、これはいらない
    # def update_qn_all_color_swap_states(self, Q, N, s):
    #     q, n = Q[s], N[s]

    #     cube = Cube()
    #     cube.state = decode_state(s)
    #     swapped_cubes = get_color_swap_states(cube)
    #     swapped_states = [encode_state(i) for i in swapped_cubes]

    #     for s_ in swapped_states:
    #         Q[s_] = q
    #         N[s_] = n

    #     return Q, N

    def squeeze_qn(self, Q, N):
        """Q, N ともに`Qのvalueの合計値が0、かつNのvalueの合計値が2以下`のkeyを削除して容量削減.
        ~~ また、Qのvalue内数値は`np.float32`, Nのvalue内数値は`np.uint32`にする. ~~ 保留

        Return:
            dict: NOT defaultdict
            dict: NOT defaultdict
        """
        key_q = np.array(list(Q.keys()))
        key_n = np.array(list(N.keys()))
        value_q = np.array(list(Q.values()))
        value_n = np.array(list(N.values()))
        sum_q = np.sum(value_q, axis=1)
        sum_n = np.sum(value_n, axis=1)

        mask = (sum_n > 2) | (sum_q != 0)

        # これを使う場合はself.Q, Nの生成時の型も指定すること
        # q = {k: np.float32(v) for k, v in zip(key_q[idx], value_q[idx])}
        # n = {k: np.uint(v) for k, v in zip(key_n[idx], value_n[idx])}
        q = dict(zip(key_q[mask], value_q[mask]))
        n = dict(zip(key_n[mask], value_n[mask]))

        # check
        for i, j in zip(q.keys(), n.keys()):
            if i != j:
                self.save_qn_file(Q, N, Q_filename=None, Q_filedir="data")
                raise Exception("saved original Q and N.")

        return q, n

    def save_qn_file(self, Q, N, Q_filename, Q_filedir):
        # Save Q & N
        Q, N = self.squeeze_qn(Q, N)
        dt = datetime.datetime.now()
        filename = ("QN_{}.pkl".format(dt.strftime("%Y%m%d%H%M"))
                    if Q_filename is None else Q_filename)
        with open(Path(Q_filedir, filename), "wb") as f:
            pickle.dump([dict(Q), dict(N)], f)
            print(f"{len(Q)=}")

    def replace_wasted_work(self, actions: np.ndarray):
        """`F`のあとに`F_`のような無駄な動きをなくす.

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


def get_newest_qn_file(dir="data/"):
    files = [i for i in Path(dir).iterdir()
             if i.name.startswith("QN") and i.name.endswith(".pkl")]
    if len(files) == 0:
        return None
    else:
        dts = [datetime.datetime.strptime(i.name, "QN_%Y%m%d%H%M.pkl")
               for i in files]
        idx = dts.index(max(dts))

        return str(files[idx])
