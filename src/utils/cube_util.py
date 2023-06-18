import numpy as np
import matplotlib.pyplot as plt

from src.env.cube import Cube, COLOR_CHARS
from src.env.action import rotate_to_home_pos, step


def show_cube(cube: Cube, home_pos=False, ax=None, fig=None):
    """
    Args:
        cube (Cube):
        home_pos (bool, optional): If True, rotate to home position.
            Defaults to False.
        ax (Axes, optional): Axes for drawing. Defaults to None.
        fig (Figure, optional): Figure for drawing. Defaults to None.
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

    if show_flag:
        plt.show()
    else:
        return ax


def get_color_swap_states(cube: Cube):
    """Return all color-swapped states from the current state.

    1つの状態を学んだときに、色だけ入れ替わった同じ状態も学習するために使う.
    """

    def get_all_color_pattern():
        """X, Y, Z を使って、現状含めすべての色の位置関係を取得する"""
        ptn = []
        actions = [
            "Y", "Y", "Y",        # WHITE TOP  : ホームポジションから見た場合
            "X", "Y", "Y", "Y",   # GREEN TOP
            "X", "Y", "Y", "Y",   # ORANGE TOP
            "Z_", "Y", "Y", "Y",  # YELLOW TOP
            "Z", "Y", "Y", "Y",   # BLUE TOP
            "Z", "Y", "Y", "Y",   # RED TOP
        ]
        c = Cube()
        ptn.append(c.state[:, 1, 1])
        for a in actions:
            step(c, a)
            ptn.append(c.state[:, 1, 1])

        return ptn

    cubes = []
    idxs = [np.argwhere(cube.state == i) for i in range(6)]
    for ptn in get_all_color_pattern():
        arr = cube.state.copy()
        for idx, color in zip(idxs, ptn):
            for x, y, z in idx:
                arr[x, y, z] = color

        c = Cube()
        c.state = arr
        if not cube == c:  # 現在と同じものは含まない
            cubes.append(c)

    return cubes


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
