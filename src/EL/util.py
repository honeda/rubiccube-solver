from src.env.cube import Cube
from src.env.action import steps
from src.utils.cube_util import show_cube
from src.env.action import int2str_actions


def save_theme_fig(scramble_actions, theme_num):
    dummy_cube = Cube()
    steps(dummy_cube, scramble_actions)

    dir_ = "data/figure"
    filename = f"/{theme_num:0>4}_{'-'.join(int2str_actions(scramble_actions))}.png"
    show_cube(dummy_cube, save=dir_ + filename)
