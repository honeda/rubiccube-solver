import math
from collections import defaultdict

import numpy as np

from src.EL.el_agent import ELAgent, ACTION_NUMS
from src.EL.util import load_qn_file, save_qn_file
from src.env.action import int2str_actions, replace_wasted_work


class MonteCarloAgent(ELAgent):

    def __init__(self, epsilon=0.1):
        super().__init__(epsilon)

    def learn(self, env, QN_file=None, n_theme=50, n_theme_step=3, n_unscramble_step=20,
              n_episode=1000, theme_actions=None, gamma=0.9, report_interval=100,
              Q_filedir="data/", Q_filename=None):
        """
        Args:
            env (Environment):
            QN_file (str, optional): Q file path. Defaults to None.
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
            report_interval (int, optional): Defaults to 100.
            Q_filedir (str, optional): Defaults to "data/".
            Q_filename (_type_, optional): Defaults to None.
        """
        # Prepare
        self.init_log()

        if QN_file:
            self.Q, self.N = load_qn_file(QN_file)
        else:
            self.Q = defaultdict(lambda: [0] * len(ACTION_NUMS))
            self.N = defaultdict(lambda: [0] * len(ACTION_NUMS))

        if theme_actions is None:
            theme_actions = np.random.choice(ACTION_NUMS, size=(n_theme_step, n_theme))
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

                    self.N[s][a] += 1  # count of s, a pair
                    alpha = 1 / self.N[s][a]
                    self.Q[s][a] += alpha * (G - self.Q[s][a])

                if e != 0 and e % report_interval == 0:
                    self.show_reward_log(episode=e)

        save_qn_file(self.Q, self.N, Q_filename, Q_filedir)

    # def calc_auto_gamma(self, n_theme_step):
    #     """手数`n_theme_step`を入れたとき0.05になる値を返す
    #     手数が大きいほど1に近い値を返す. 最大手数は30を想定.

    #     不要と判断し削除.
    #     これを使うと短手数でgammaが低くなるが、
    #     そうすると最短手数のアクションの報酬も低くなりがち.
    #     その状態で長手数(gmmmaが高い)を解かせたときに、epsilonにより
    #     最短手数でないアクションが選択されたときにその手の報酬が
    #     高く評価され(gammaが高いため) 最短手数の報酬を超える可能性がある
    #     と考えたため.

    #     """
    #     min_ = 0.05
    #     gamma = min_ ** (1 / n_theme_step)

    #     return gamma

    # X, Y, Z を使わない場合、これはいらない
    # def update_qn_all_color_swap_states(self, Q, self.N, s):
    #     q, n = Q[s], N[s]

    #     cube = Cube()
    #     cube.state = decode_state(s)
    #     swapped_cubes = get_color_swap_states(cube)
    #     swapped_states = [encode_state(i) for i in swapped_cubes]

    #     for s_ in swapped_states:
    #         Q[s_] = q
    #         N[s_] = n

    #     return Q, N
