import numpy as np

from src.env import basic_action as ba  # noqa F401
from src.env.cube import (
    Cube,
    TOP, LEFT, BACK, RIGHT, FRONT, UNDER,
    SURFACE_CHARS,
)


ACTION_CHARS = "F F_ U U_ R R_ D D_ B B_ L L_".split()
ACTION_NUMS = list(range(len(ACTION_CHARS)))

ACTION_CHARS_WITH_ROTATE = ACTION_CHARS + "X X_ Y Y_ Z Z_".split()
ACTION_NUMS_WITH_ROTATE = list(range(len(ACTION_CHARS_WITH_ROTATE)))

ACTION_FUNCS = [eval(f"ba.{i}") for i in ACTION_CHARS_WITH_ROTATE]


def step_by_str(cube: Cube, action_str):
    idx = ACTION_CHARS_WITH_ROTATE.index(action_str)
    ACTION_FUNCS[idx](cube)


def step_by_int(cube: Cube, action_int):
    ACTION_FUNCS[action_int](cube)


def step(cube: Cube, action):
    # これだと`numpy.str_`のとき検知できない
    # if type(action) == str:
    try:
        step_by_str(cube, action)
    except ValueError:
        step_by_int(cube, action)


def steps(cube: Cube, actions):
    for a in actions:
        step(cube, a)


def str2int_actions(actions):
    return [ACTION_CHARS_WITH_ROTATE.index(i) for i in actions]


def int2str_actions(actions):
    return [ACTION_CHARS_WITH_ROTATE[i] for i in actions]


def replace_wasted_work(actions: np.ndarray):
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


def get_reverse_actions(actions, return_type="int"):
    # これだと`numpy.str_`のとき検知できない
    # if type(actions[0]) != str:

    try:
        int(actions[0])
        actions = int2str_actions(actions)
    except ValueError:
        pass

    lst = []
    for a in actions[::-1]:
        if a.endswith("_"):
            lst.append(a[0])
        else:
            lst.append(a + "_")

    if return_type == "str":
        return lst
    elif return_type == "int":
        return str2int_actions(lst)
    else:
        raise Exception


def rotate_to_home_pos(cube: Cube, get_rotate_actions=False):
    """Rotate so that white is top face and red is front face.
    """

    # White face to the top
    w_idx, _ = cube.current_wg_pos
    if w_idx != TOP:
        if w_idx == LEFT:
            step(cube, "Z")
        elif w_idx == BACK:
            step(cube, "X_")
        elif w_idx == RIGHT:
            step(cube, "Z_")
        elif w_idx == FRONT:
            step(cube, "X")
        elif w_idx == UNDER:
            step(cube, "X")
            step(cube, "X")
        else:
            raise Exception

    # Green face to the front
    _, g_idx = cube.current_wg_pos
    if g_idx != FRONT:
        if g_idx == LEFT:
            step(cube, "Y_")
        elif g_idx == BACK:
            step(cube, "Y")
            step(cube, "Y")
        elif g_idx == RIGHT:
            step(cube, "Y")
        else:
            raise Exception(f"Green face is {SURFACE_CHARS[g_idx]}")
