import numpy as np
import matplotlib.pyplot as plt

from src.env.cube import Cube, COLOR_CHARS
from src.env.action import rotate_to_home_pos, step


def show_cube(cube: Cube, home_pos=False, ax=None, fig=None, save=""):
    """
    Args:
        cube (Cube):
        home_pos (bool, optional): If True, rotate to home position.
            Defaults to False.
        ax (Axes, optional): Axes for drawing. Defaults to None.
        fig (Figure, optional): Figure for drawing. Defaults to None.
        save (str, optional): save file name. Defaults to "".
    """
    c = cube.copy()
    if home_pos:
        rotate_to_home_pos(c)

    show_flag = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 4))
        show_flag = True

    points = [[5, 5], [2, 5], [5, 8], [8, 5], [5, 2], [11, 5]]
    for arr, (X, Y) in zip(c.state, points):
        arr = arr[::-1]  # 描画用に反転
        for i in range(3):
            for j in range(3):
                x_range = [X + j, X + j + 1]
                y_range = [Y + i, Y + i + 1]
                ax.fill_between(x_range, *y_range,
                                color=COLOR_CHARS[arr[i, j]].lower())
                # print(f"{i=}, {x_range=}, {j=}, {y_range}")
                # draw lines
                ax.vlines(x_range, *y_range, color="k", linewidth=.75)
                ax.hlines(y_range, *x_range, color="k", linewidth=.75)

    ax.set_xlim(0, 16)
    ax.set_ylim(1, 12)
    ax.set_aspect(1.05)
    ax.axis("off")
    if fig:
        fig.patch.set_facecolor("lavender")

    if show_flag and save == "":
        plt.show()
    elif save != "":
        plt.savefig(save)
        plt.close()
    else:
        return ax


# def get_color_swap_states(cube: Cube):
#     """Return all color-swapped states from the current state.

#     1つの状態を学んだときに、色だけ入れ替わった同じ状態も学習するために使う.
#     """

#     def get_all_color_swap_pattern():
#         """X, Y, Z を使って、現状含めすべての色の位置関係を取得する"""
#         ptn = []

#         # ホームポジション(wwite top, red front)に固定してスワップするのに必要な位置
#         actions = [["Y"], ["Y"], ["Y"], ["Z"], ["Z", "Z"]]
#         c = Cube()
#         ptn.append(c.state[:, 1, 1])
#         for a in actions:
#             steps(c, a)
#             ptn.append(c.state[:, 1, 1])

#         return ptn

#     cubes = []
#     idxs = [np.argwhere(cube.state == i) for i in range(6)]
#     for ptn in get_all_color_swap_pattern():
#         arr = cube.state.copy()
#         for idx, color in zip(idxs, ptn):
#             for x, y, z in idx:
#                 arr[x, y, z] = color

#         c = Cube()
#         c.state = arr
#         rotate_to_home_pos(c)  # ホームポジションに戻す
#         if not cube == c:
#             cubes.append(c)

#     return cubes


def get_color_swap_states(cube: Cube, debug=False):
    """Return all color-swapped states from the current state.

    1つの状態を学んだときに、色だけ入れ替わった同じ状態も学習するために使う.

    Args:
        cube (Cube):
        debug (bool, optional): Trueの場合重複する局面も含めすべて返す
    Return:
        cubes (list): [Cube, Cube, ...]
        translated_actions: [dict, dict, ...]
    """

    def get_all_color_pattern():
        """X, Y, Z を使って、現状含めすべての色の位置関係を取得する"""
        rotate_actions = [
            # 全方向確かめる必要あり！
            # 節約しようとすると`get_color_swap_states_with_action()`
            # で"U"や”D"の操作の局面がうまく見つけられない
            "", "Y", "Y", "Y",        # WHITE TOP  : ホームポジションから見た場合
            "X", "Y", "Y", "Y",   # GREEN TOP
            "X", "Y", "Y", "Y",   # ORANGE TOP
            "Z_", "Y", "Y", "Y",  # YELLOW TOP
            "Z", "Y", "Y", "Y",   # BLUE TOP
            "Z", "Y", "Y", "Y",   # RED TOP
        ]
        action_transitions = [
            # i番目の要素はrotate_actionのi番目の要素と対応
            # rotateしたときの向きでのアクションとホームポジションでのアクションの対応関係
            {"F": "F", "U": "U", "R": "R", "D": "D", "B": "B", "L": "L"},  # 何もしてないとき
            {"F": "R", "U": "U", "R": "B", "D": "D", "B": "L", "L": "F"},  # Yしたときの対応
            {"F": "B", "U": "U", "R": "L", "D": "D", "B": "F", "L": "R"},  # さらにYしたときの対応
            {"F": "L", "U": "U", "R": "F", "D": "D", "B": "R", "L": "B"},  # さらにY
            {"F": "D", "U": "L", "R": "F", "D": "R", "B": "U", "L": "B"},  # さらにX
            {"F": "F", "U": "L", "R": "U", "D": "R", "B": "B", "L": "D"},  # Y
            {"F": "U", "U": "L", "R": "B", "D": "R", "B": "D", "L": "F"},  # Y
            {"F": "B", "U": "L", "R": "D", "D": "R", "B": "F", "L": "U"},  # Y
            {"F": "R", "U": "B", "R": "D", "D": "F", "B": "L", "L": "U"},  # X
            {"F": "D", "U": "B", "R": "L", "D": "F", "B": "U", "L": "R"},  # Y
            {"F": "L", "U": "B", "R": "U", "D": "F", "B": "R", "L": "D"},  # Y
            {"F": "U", "U": "B", "R": "R", "D": "F", "B": "D", "L": "L"},  # Y
            {"F": "U", "U": "R", "R": "B", "D": "L", "B": "D", "L": "F"},  # Z_
            {"F": "F", "U": "R", "R": "D", "D": "L", "B": "B", "L": "U"},  # Y
            {"F": "D", "U": "R", "R": "B", "D": "L", "B": "U", "L": "F"},  # Y
            {"F": "B", "U": "R", "R": "U", "D": "L", "B": "F", "L": "D"},  # Y
            {"F": "B", "U": "D", "R": "R", "D": "U", "B": "F", "L": "L"},  # Z
            {"F": "R", "U": "D", "R": "B", "D": "U", "B": "L", "L": "F"},  # Y
            {"F": "F", "U": "D", "R": "L", "D": "U", "B": "B", "L": "R"},  # Y
            {"F": "L", "U": "D", "R": "B", "D": "U", "B": "R", "L": "F"},  # Y
            {"F": "L", "U": "F", "R": "D", "D": "B", "B": "R", "L": "U"},  # Z
            {"F": "D", "U": "F", "R": "R", "D": "B", "B": "U", "L": "L"},  # Y
            {"F": "R", "U": "F", "R": "B", "D": "B", "B": "L", "L": "U"},  # Y
            {"F": "U", "U": "F", "R": "L", "D": "B", "B": "D", "L": "R"}   # Y
        ]
        # 逆回転のアクションを追加
        for d in action_transitions:
            for k, v in zip(list(d.keys()), list(d.values())):
                d[k + "_"] = v + "_"

        ptn = []
        translated_actions = []

        c = Cube()
        for i, ra in enumerate(rotate_actions):
            if ra != "":
                step(c, ra)
            ptn.append(c.state[:, 1, 1])
            translated_actions.append(action_transitions[i])

        return ptn, translated_actions

    cubes = []
    traslated_actions = []
    curr_color_masks = [cube.state == i for i in range(6)]
    for ptn, trans_a in zip(*get_all_color_pattern()):
        arr = cube.state.copy()
        for mask, color in zip(curr_color_masks, ptn):
            arr[mask] = color
            # for x, y, z in idx:
            #     arr[x, y, z] = color

        c = Cube()
        c.state = arr
        rotate_to_home_pos(c)
        if debug:
            cubes.append(c)
            traslated_actions.append(trans_a)
        else:
            flag = np.all([c != i for i in cubes])
            if (cube != c) and flag:  # 現在と同じものは含まない
                cubes.append(c)
                traslated_actions.append(trans_a)

    return cubes, traslated_actions


def encode_state(cube: Cube):
    """Convert to integer representing the current state.

    キューブの状態(cube.state)は`np.ndarray`だがこのままでは保存容量が大きいので`int`にする.
    なお、cube.state配列が`0`始まりのときのために先頭にダミーの1をつけている.
    """
    txt = str(cube.state.ravel()).replace("\n", "").replace(" ", "")[1: -1]

    return int("1" + txt)


def decode_state(encoded: int):
    txt = str(encoded)[1:]  # 先頭の"1"を除去
    txt = "[" + ",".join(txt) + "]"

    return np.array(eval(txt)).reshape(6, 3, 3)
