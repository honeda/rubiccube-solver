import numpy as np
import matplotlib.pyplot as plt

from src.env.cube import Cube, COLOR_CHARS
from src.env.action import rotate_to_home_pos


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
    # これが世界基準だがアクションも変えないといけない
    # points = [[5, 8], [2, 5], [11, 5], [8, 5], [5, 5], [5, 2]]
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


def encode_state(cube: Cube):
    """Convert to integer representing the current state.

    キューブの状態(cube.state)は`np.ndarray`だがこのままでは保存容量が大きいので`int`にする.
    なお、cube.state配列が`0`始まりのときのために先頭にダミーの1をつけている.
    """
    s = cube.state.ravel()
    encoded = int("1" + "".join([str(i) for i in s]))

    return encoded


def decode_state(encoded: int):
    txt = str(encoded)[1:]  # 先頭の"1"を除去
    txt = "[" + ",".join(txt) + "]"

    return np.array(eval(txt)).reshape(6, 3, 3)
