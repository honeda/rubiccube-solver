import datetime
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np

from src.env.cube import Cube
from src.env.action import steps
from src.utils.cube_util import show_cube
from src.env.action import int2str_actions
from src.EL.el_agent import ACTION_NUMS


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


def squeeze_qn(Q, N):
    """Q, N ともに`Qのvalueの合計値が0のkeyを削除して容量削減.

    成功したことないstateの場合、一様分布からアクションを決めるため
    各アクションの試行回数に偏りはない(はず）.
    よって成功したことのないstateの試行回数(N)を保存しておく必要はない.

    Return:
        dict: NOT defaultdict
        dict: NOT defaultdict
    """
    key_q = np.array(list(Q.keys()))
    key_n = np.array(list(N.keys()))
    value_q = np.array(list(Q.values()))
    value_n = np.array(list(N.values()))
    sum_q = np.sum(value_q, axis=1)

    mask = (sum_q != 0)
    q = dict(zip(key_q[mask], value_q[mask]))
    n = dict(zip(key_n[mask], value_n[mask]))

    # check
    for i, j in zip(q.keys(), n.keys()):
        if i != j:
            raise Exception("saved original Q and N.")

    return q, n


def save_qn_file(Q, N, Q_filename, Q_filedir):
    # Save Q & N
    Q, N = squeeze_qn(Q, N)
    dt = datetime.datetime.now()
    filename = ("QN_{}.pkl".format(dt.strftime("%Y%m%d%H%M"))
                if Q_filename is None else Q_filename)
    with open(Path(Q_filedir, filename), "wb") as f:
        pickle.dump([dict(Q), dict(N)], f)
        print(f"{len(Q)=}")


def load_qn_file(file_path):
    """
    Args:
        file_path (str): file path

    Returns:
        Q (defaultdict)
        N (defaultdict)
    """
    Q = defaultdict(lambda: [0] * len(ACTION_NUMS))
    N = defaultdict(lambda: [0] * len(ACTION_NUMS))

    Q_, N_ = pickle.load(open(file_path, "rb"))
    # dict -> defaultdict
    for k, v in Q_.items():
        Q[k] = v
    for k, v in N_.items():
        N[k] = v
    print(f"{len(Q)=}, {len(N)=}")

    return Q, N


def save_theme_fig(scramble_actions, theme_num):
    dummy_cube = Cube()
    steps(dummy_cube, scramble_actions)

    dir_ = "data/figure"
    filename = f"/{theme_num:0>4}_{'-'.join(int2str_actions(scramble_actions))}.png"
    show_cube(dummy_cube, save=dir_ + filename)
