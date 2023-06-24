import numpy as np

from src.env.cube import Cube
from src.env.action import rotate_to_home_pos, step, str2int_actions, ACTION_CHARS


ROTATE_ACTIONS = [
    # 全方向のキューブが取得できるよう回転させるためのアクション
    "", "Y", "Y", "Y",    # WHITE TOP  : ホームポジションから見た場合
    "X", "Y", "Y", "Y",   # GREEN TOP
    "X", "Y", "Y", "Y",   # ORANGE TOP
    "Z_", "Y", "Y", "Y",  # YELLOW TOP
    "Z", "Y", "Y", "Y",   # BLUE TOP
    "Z", "Y", "Y", "Y",   # RED TOP
]

def define_action_transitions():
    action_transitions_str = [  # 人間がみやすいように定義
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
        {"F": "U", "U": "R", "R": "F", "D": "L", "B": "D", "L": "B"},  # Z_
        {"F": "F", "U": "R", "R": "D", "D": "L", "B": "B", "L": "U"},  # Y
        {"F": "D", "U": "R", "R": "B", "D": "L", "B": "U", "L": "F"},  # Y
        {"F": "B", "U": "R", "R": "U", "D": "L", "B": "F", "L": "D"},  # Y
        {"F": "B", "U": "D", "R": "R", "D": "U", "B": "F", "L": "L"},  # Z
        {"F": "R", "U": "D", "R": "F", "D": "U", "B": "L", "L": "B"},  # Y
        {"F": "F", "U": "D", "R": "L", "D": "U", "B": "B", "L": "R"},  # Y
        {"F": "L", "U": "D", "R": "B", "D": "U", "B": "R", "L": "F"},  # Y
        {"F": "L", "U": "F", "R": "D", "D": "B", "B": "R", "L": "U"},  # Z
        {"F": "D", "U": "F", "R": "R", "D": "B", "B": "U", "L": "L"},  # Y
        {"F": "R", "U": "F", "R": "U", "D": "B", "B": "L", "L": "D"},  # Y
        {"F": "U", "U": "F", "R": "L", "D": "B", "B": "D", "L": "R"}   # Y
    ]
    # 逆回転のアクションを追加
    for d in action_transitions_str:
        for k, v in zip(list(d.keys()), list(d.values())):
            d[k + "_"] = v + "_"

    # 速度のためdictから変更後のアクションの番号listに変換する.
    # 変更前のアクションはリストのインデックスで分かる.
    tmp = []
    for dic in action_transitions_str:
        tmp.append([dic[i] for i in ACTION_CHARS])

    action_transitions_int = [str2int_actions(i) for i in tmp]

    return action_transitions_int


ACTION_TRANSITIONS = define_action_transitions()


def _get_all_color_pattern():
    """X, Y, Z を使って、現状含めすべての色の位置関係を取得する"""
    ptn = []
    translated_actions = []

    c = Cube()
    for i, ra in enumerate(ROTATE_ACTIONS):
        if ra != "":
            step(c, ra)
        ptn.append(c.state[:, 1, 1])
        translated_actions.append(ACTION_TRANSITIONS[i])

    return ptn, translated_actions


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
    cubes = []
    traslated_actions = []
    curr_color_masks = [cube.state == i for i in range(6)]
    for ptn, trans_a in zip(*_get_all_color_pattern()):
        arr = cube.state.copy()
        for mask, color in zip(curr_color_masks, ptn):
            arr[mask] = color

        c = Cube()
        c.state = arr
        rotate_to_home_pos(c)
        if debug:
            cubes.append(c)
            traslated_actions.append(trans_a)
        else:
            flag = np.all([c != i for i in cubes])
            if (cube != c) and flag:  # 現在と同じものとすでにあるものは含まない
                cubes.append(c)
                traslated_actions.append(trans_a)

    return cubes, traslated_actions
