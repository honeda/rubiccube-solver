import math
from collections import defaultdict

from src.EL.el_agent import ELAgent
from src.env.action import int2str_actions, generate_random_action, ACTION_NUMS
from src.utils.cube_util import encode_state


class MonteCarloAgent(ELAgent):

    def __init__(self, epsilon=0.1):
        super().__init__(epsilon)

    def learn(self, env, Q_file=None, n_theme=50, n_theme_step=3, n_unscramble_step="auto",
              n_episode=1000, theme_actions=None, checkpoint_interval=200, report_interval=100,
              Q_filedir="data/EL/monte_carlo", Q_filename=None):
        """
        Args:
            env (Environment):
            Q_file (str, optional): Q file path. Defaults to None.
            n_theme (int, optional): Num of theme. Defaults to 50.
            n_theme_step (int, optional): Num of step each theme.
                Defaults to 3.
            n_unscramble_step (int or str, optional): Num of unscramble step.
                When "auto", this argument to be value of `n_theme_step`.
                Defaults to "auto".
            n_episode (int, optional): Num of episode per theme.
                Defaults to 1000.
            theme_actions (list, optional): Use when you want to give
                a specific theme. ex) [["F", "B"], ["D", "F_", "B"].
                `n_theme` and `theme_steps` are ignored when this argument
                is not None. Defaults to None.
            checkpoint_interval (int, optional): Defaults to 100.
            report_interval (int, optional): Defaults to 100.
            Q_filedir (str, optional): Defaults to "data/".
            Q_filename (_type_, optional): Defaults to None.
        """
        gamma = 0.9  # FIX

        if n_unscramble_step == "auto":
            n_unscramble_step = n_theme_step

        # Prepare
        self.init_log()

        if Q_file:
            self.Q = self.load_q_file(Q_file)
        else:
            self.Q = defaultdict(lambda: [0] * len(ACTION_NUMS))

        N = defaultdict(lambda: [0] * len(ACTION_NUMS))

        if theme_actions is None:
            theme_actions = [generate_random_action(n_theme_step) for _ in range(n_theme)]

        # Learning
        self.logger.info(f"Start learning. {gamma=:.2f}, {n_theme=}, {n_theme_step=}"
                         f", {n_unscramble_step=}")
        appeared_states = []
        never_done_states = []
        for theme, scramble_actions in enumerate(theme_actions, 1):
            # Scramble
            print("==============================================================")
            print(f"No.{theme:0>4} Theme scene: {int2str_actions(scramble_actions)}")

            env.set_game_start_position(scramble_actions)

            # スクランブルした状態が完成状態なら次へ
            if env.cube.is_solved:
                continue

            done_th = n_episode / 1  # この回数完成させないと次のthemeにいかない
            e_max = n_episode * 7  # done_thに達していなくてもこのエピソード数で次へ
            n_done = 0
            e = 0

            # Play episodes
            while (n_done < done_th) or (e < n_episode):
                e += 1
                env.reset_to_gamestart()
                prev_action = None
                s = env.states
                done = False
                experience = []

                for _ in range(n_unscramble_step):
                    a = self.policy(s, prev_action, ACTION_NUMS)
                    prev_action = int(a)
                    n_state, reward, done = env.step(a)

                    experience.append({"state": s, "action": a, "reward": reward})
                    s = n_state
                    if done:
                        n_done += 1
                        break

                if e == e_max:
                    if n_done == 0:
                        never_done_states.append(scramble_actions)

                        # 1回も解けなかったstateなのに価値がある場合、color-swapのバグかもしれない
                        env.reset_to_gamestart()
                        state = encode_state(env.cube)
                        if sum(self.Q[state]) != 0:
                            self.save_q_file("problem_Q.pkl", Q_filedir)
                            msg = ("Maybe bug in the color swap logic. "
                                   f"{scramble_actions=}, {state=}, {self.Q[state]=}")
                            self.logger.error(msg)
                            raise Exception
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
                    values = self.Q[s]  # アクセス回数削減
                    if G > values[a]:
                        # 最短手数しばりを求める限り一度見つけた価値が下げる必要はない？
                        # 価値が下げられるのはより長い手を見つけたとき（gammaの手数ペナルティを受けるため）
                        # より長い手での完成により価値を下げる必要はないはず.
                        values[a] += alpha * (G - values[a])

                    if sum(self.Q[s]) != 0:
                        appeared_states.append(s)

                if e != 0 and e % report_interval == 0:
                    self.show_reward_log(episode=e)

            if theme != 0 and theme % checkpoint_interval == 0:
                self.logger.info(f"Checkpoint. Current theme number is {theme}.")
                self.checkpoint(appeared_states, Q_filename, Q_filedir)
                appeared_states = []

        # Post process
        print("Never done states:")
        for s in never_done_states:
            print(s)
        self.logger.info(f"Finish. {n_theme=}, {len(never_done_states)=}"
                         f" ({(len(never_done_states) / n_theme) *100:.1f}%)")
        self.checkpoint(appeared_states, Q_filename, Q_filedir)
