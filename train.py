import argparse

from src.env import Environment
from src.EL.monte_carlo import MonteCarloAgent
from src.EL.util import get_newest_qn_file


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--QN_file", default=None)
    parser.add_argument("--n_theme_step", default=3, type=int)
    parser.add_argument("--n_unscramble_step", default=20, type=int)
    parser.add_argument("--n_theme", default=50, type=int)
    parser.add_argument("--n_episode", default=1000, type=int)
    parser.add_argument("--report_interval", default=100, type=int)
    parser.add_argument("--gamma", default=0.9, type=int)
    parser.add_argument("--epsilon", default=0.1, type=float)

    args = parser.parse_args()

    if args.QN_file is None:
        QN_file = get_newest_qn_file()
        print(f"{QN_file=}")

    agent = MonteCarloAgent(epsilon=args.epsilon)
    env = Environment()
    agent.learn(
        env,
        n_theme_step=args.n_theme_step,
        n_theme=args.n_theme,
        n_unscramble_step=args.n_unscramble_step,
        n_episode=args.n_episode,
        QN_file=QN_file,
        report_interval=args.report_interval,
        gamma=args.gamma
    )
    agent.show_reward_log()
