import numpy as np
import matplotlib.pyplot as plt

from src.env.cube import Cube, COLOR_CHARS
from src.env.action import rotate_to_home_pos


def show_cube(cube: Cube, home_pos=False):
    """
    Args:
        cube (Cube):
        home_pos (bool, optional): if True, rotate to home position.
            Defaults to False.
    """
    c = cube.copy()
    if home_pos:
        rotate_to_home_pos(c)

    fig, ax = plt.subplots(figsize=(6, 4))
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
    ax.axis("off")
    fig.patch.set_facecolor("lavender")
    plt.show()


def encode_state(cube: Cube):
    """キューブの状態を表す数値(int)に変換する.

    キューブの状態(cube.state)は`np.ndarray`だがこのままでは保存容量が大きいので`int`にする.
    なお、cube.state配列が`0`始まりのときのために先頭にダミーの1をつけている.

    Args:
        cube (Cube):
    """
    txt = str(cube.state.ravel()).replace("\n", "").replace(" ", "")[1: -1]

    return int("1" + txt)


def decode_state(encoded: int):
    txt = str(encoded)[1:]  # 先頭の"1"を除去
    txt = "[" + ",".join(txt) + "]"

    return np.array(eval(txt)).reshape(6, 3, 3)
