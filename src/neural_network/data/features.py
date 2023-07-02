import json
import numpy as np

GAMMA = json.load(open("config/global_parameters.json"))["gamma"]


def make_value(action):
    """For value network output.

    Args:
        values (list): Values of Q-table.
    Return:
        float
    """
    return np.max(action)


def make_multi_label(action):
    """For output of MULTI-LABEL policy network.

    Args:
        values (list): Values of Q-table.
    Return:
        np.ndarray: Shape is (len(ACTION_NUMS),)
    """
    # 最短手数が2手の場合、価値は最大で gamma**(2-1) = 0.9**1 = 0.9
    # 最短手数が3手の場合、価値は最大で gamma**(3-1) = 0.81
    # 3手かかる場合、価値が0.81以上になることはない.
    # つまり 0.81 < x <= 0.9 のアクションを選べば2手で終わるということになるので
    # この範囲のアクションがいくつかある場合はすべて2手で終わるということになる.

    bins = np.array([GAMMA ** i for i in range(50)])
    idx = np.digitize(np.max(action), bins)

    label = ((action <= bins[idx] + 1e6) & (action > bins[idx + 1])).astype(float)
    if sum(label) == 0:
        msg = "Label preprocessing error."
        raise Exception(msg)

    return label
