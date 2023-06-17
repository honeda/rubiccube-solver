from src.env import basic_action as ba  # noqa F401
from src.env.cube import (
    Cube,
    TOP, LEFT, BACK, RIGHT, FRONT, UNDER,
    SURFACE_CHARS,
)


ACTION_CHARS = "X X_ Y Y_ Z Z_ F F_ R R_".split()


def step_by_str(cube: Cube, action_str):
    eval(f"ba.{action_str}")(cube)


def step_by_int(cube: Cube, action_int):
    eval(f"ba.{ACTION_CHARS[action_int]}")(cube)


def step(cube: Cube, action):
    # これだと`numpy.str_`のとき検知できない
    # if type(action) == str:
    try:
        step_by_str(cube, action)
    except SyntaxError:
        step_by_int(cube, action)


def steps(cube: Cube, actions):
    for a in actions:
        step(cube, a)


def str2int_actions(actions):
    return [ACTION_CHARS.index(i) for i in actions]


def int2str_actions(actions):
    return [ACTION_CHARS[i] for i in actions]


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


def rotate_to_home_pos(cube: Cube):
    """Rotate so that white is top face and red is front face.
    """

    # White face to the top
    w_idx, _ = cube.current_wr_pos
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

    # Red face to the front
    _, r_idx = cube.current_wr_pos
    if r_idx != FRONT:
        if r_idx == LEFT:
            step(cube, "Y_")
        elif r_idx == BACK:
            step(cube, "Y")
            step(cube, "Y")
        elif r_idx == RIGHT:
            step(cube, "Y")
        else:
            raise Exception(f"RED face is {SURFACE_CHARS[r_idx]}")
