import argparse

from src.env import Environment
from src.EL.monte_carlo import MonteCarloAgent, get_newest_qn_file
from src.EL.q_learning import QLearningAgent, get_newest_q_file


def monte_carlo(args):

    if args.Q_file is None:
        Q_file = get_newest_qn_file()
        print(f"{Q_file=}")

    agent = MonteCarloAgent(epsilon=args.epsilon)
    agent.learn(
        env,
        n_theme_step=args.n_theme_step,
        n_theme=args.n_theme,
        n_unscramble_step=args.n_unscramble_step,
        n_episode=args.n_episode,
        QN_file=Q_file,
        report_interval=args.report_interval,
        gamma=args.gamma
    )

    agent.show_reward_log()


def q_learning(args):

    if args.Q_file is None:
        Q_file = get_newest_q_file()
        print(f"{Q_file=}")

    agent = QLearningAgent(epsilon=args.epsilon)
    agent.learn(
        env,
        n_theme_step=args.n_theme_step,
        n_theme=args.n_theme,
        n_unscramble_step=args.n_unscramble_step,
        n_episode=args.n_episode,
        Q_file=Q_file,
        report_interval=args.report_interval,
        gamma=args.gamma,
        learning_rate=args.learning_rate
    )

    agent.show_reward_log()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", "-m", type=str, help="'monte_carlo' or 'Q_learning'.")
    parser.add_argument("--Q_file", default=None)
    parser.add_argument("--n_theme_step", default=3, type=int)
    parser.add_argument("--n_unscramble_step", default=20, type=int)
    parser.add_argument("--n_theme", default=50, type=int)
    parser.add_argument("--n_episode", default=1000, type=int)
    parser.add_argument("--report_interval", default=100, type=int)
    parser.add_argument("--gamma", default=0.9, type=int)
    parser.add_argument("--learning_rate", default=0.1, type=int)
    parser.add_argument("--epsilon", default=0.1, type=float)

    args = parser.parse_args()

    env = Environment()

    if args.mode == "monte_carlo":
        monte_carlo(args)
    elif args.mode == "Q_learning":
        q_learning(args)
    else:
        raise Exception
