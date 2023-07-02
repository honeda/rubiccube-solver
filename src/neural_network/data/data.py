import json
import numpy as np

from src.env.action import ACTION_NUMS
from src.utils.cube_util import decode_state

GAMMA = json.load(open("config/global_parameters.json"))["gamma"]

dtypeState = np.dtype((np.uint8, 6 * 3 * 3))
dtypeAction = np.dtype((np.float32, len(ACTION_NUMS)))

QTable = np.dtype([
    ("state", dtypeState),
    ("action", dtypeAction)
])


def dict2ndarray(Q):
    """Convert a Q-table of type dict to a type-specified np.ndarray.

    Args:
        Q (dict): Q-table
    Return:
        np.ndarray
    """
    Q_arr = np.zeros(len(Q), dtype=QTable)
    Q_arr["state"] = [decode_state(i).ravel() for i in Q.keys()]
    Q_arr["action"] = list(Q.values())

    return Q_arr


def remove_low_value_rows(data):
    th = GAMMA ** 10  # 30手かかる局面の価値より小さいデータを除去する.
    mask = (np.max(data["action"], axis=1) > th)
    data = data[mask]

    return data
