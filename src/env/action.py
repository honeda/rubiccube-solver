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


def generate_random_action(length, return_type="int"):
    """`F->F_`のような無駄な操作を排除した任意の長さのアクションlistを返す.

    排除するのは以下のようなアクションの組み合わせ.
    * F -> F_
    * F -> F -> F  (= F_)
    * F -> B -> F_ (= B)
    * F -> F -> B -> F  (= F_ -> B)
    * F -> B -> F -> F  (= F_ -> B)
    * F -> B -> B -> F_ (= B -> B)
    * F -> F -> B -> B -> F (= F_ -> B -> B)
    * F -> B -> B -> F -> F (= F_ -> B -> B)

    Args:
        length (_type_): _description_
        return_type (str, optional): _description_. Defaults to "int".
    Returns:
        list: action list
    """
    opposite_face_actions = [
        # 反対の面の操作
        (8, 9),    # F: (B, B_)
        (8, 9),    # F_: (B, B_)
        (6, 7),    # U: (D, D_)
        (6, 7),    # U_: (D, D_)
        (10, 11),  # R: (L, L_)
        (10, 11),  # R_: (L, L_)
        (2, 3),    # D: (U, U_)
        (2, 3),    # D_: (U, U_)
        (0, 1),    # B: (F, F_)
        (0, 1),    # B_: (F, F_)
        (4, 5),    # L: (R, R_)
        (4, 5),    # L_: (R, R_)
    ]

    actions = []
    for _ in range(length):
        action_space = [i for i in ACTION_NUMS]
        removes = []
        if len(actions) > 0:
            # F->F_, F_->F のようになるアクションを除去
            if actions[-1] % 2 == 0:
                removes.append(actions[-1] + 1)
            else:
                removes.append(actions[-1] - 1)

        if len(actions) >= 2:

            p1 = actions[-1]
            p2 = actions[-2]
            # F->F ときているとき Fを除去
            if p1 == p2:
                removes.append(p1)

            # F->B ときているとき F_を除去
            if p1 in opposite_face_actions[p2]:
                if p2 % 2 == 0:
                    removes.append(p2 + 1)
                else:
                    removes.append(p2 - 1)

            if len(actions) >= 3:
                p3 = actions[-3]

                # F->F->B ときているとき Fを除去
                if (
                    p2 == p3
                    and p1 in opposite_face_actions[p2]
                ):
                    removes.append(p2)
                    # debug message
                    # print(int2str_actions(actions))
                    # print(int2str_actions([removes[-1]]))

                # F->B->F ときているとき Fを除去
                if (
                    p1 == p3
                    and p2 in opposite_face_actions[p1]
                ):
                    removes.append(p1)

                # F->B->B ときているとき F_を除去
                if (
                    p1 in opposite_face_actions[p3]
                    and p2 in opposite_face_actions[p3]
                ):
                    if p3 % 2 == 0:
                        removes.append(p3 + 1)
                    else:
                        removes.append(p3 - 1)

            if len(actions) >= 4:
                p4 = actions[-4]

                # F->F->B->B ときているとき Fを除去
                if (
                    p3 == p4
                    and p1 in opposite_face_actions[p3]
                    and p2 in opposite_face_actions[p3]
                ):
                    removes.append(p3)

                # F->B->B->F ときているとき Fを除去
                if (
                    p1 == p4
                    and p2 in opposite_face_actions[p1]
                    and p3 in opposite_face_actions[p1]
                ):
                    removes.append(p1)

        for i in np.unique(removes):
            action_space.remove(i)

        actions.append(np.random.choice(action_space))

    if return_type == "int":
        return actions
    else:
        return int2str_actions(actions)
