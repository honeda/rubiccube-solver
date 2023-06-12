import datetime
import math
import pickle
from pathlib import Path
from collections import defaultdict

from src.EL.el_agent import ELAgent
from src.env import Environment, ACTIONS

import numpy as np


class MonteCarloAgent(ELAgent):

    def __init__(self, epsilon=0.1, Q_file=None):
        super().__init__(epsilon)
        if Q_file:
            Q = pickle.load(open(Q_file, "rb"))
            # dict -> defaultdict
            for k, v in Q.items():
                self.Q[k] = v
            # self.Q = json.load(open(Q_file))
            print(f"{len(self.Q)=}")

    def learn(self, env, n_theme=50, theme_steps=3, theme_actions=None, n_episode=1000,
              gamma=0.9, report_interval=50, Q_filedir="data/", Q_filename=None):
        """
        Args:
            env (Environment):
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
            report_interval (int, optional): Defaults to 50.
            Q_filedir (str, optional): Defaults to "data/".
            Q_filename (_type_, optional): Defaults to None.
        """
        self.init_log()
        action_nums = list(range(len(ACTIONS)))
        self.Q = defaultdict(lambda: [0] * len(action_nums))
        N = defaultdict(lambda: [0] * len(action_nums))
        n_unscrable_step = 25

        if theme_actions is None:
            theme_actions = [np.random.choice(ACTIONS, size=theme_steps)
                             for _ in n_theme]

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
                else:
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

        # Save Q
        dt = datetime.datetime.now()
        filename = ("Q_{}.pkl".format(dt.strftime("%Y%m%d%H%M"))
                    if Q_filename is None else Q_filename)
        with open(Path(Q_filedir, filename), "wb") as f:
            pickle.dump(dict(self.Q), f)
            # json.dump(self.Q, f, indent=2)


def train():
    agent = MonteCarloAgent(epsilon=0.1, Q_file=None)
    env = Environment()
    agent.learn(env, n_episode=1_000_000, report_interval=500, Q_filedir=".")
    agent.show_reward_log()


if __name__ == "__main__":
    train()
