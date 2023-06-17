import numpy as np


COLORS = (
    WHITE,
    GREEN,
    ORANGE,
    YELLOW,
    RED,
    BLUE
) = list(range(6))

SURFACES = (
    TOP,
    LEFT,
    BACK,
    RIGHT,
    FRONT,
    UNDER
) = list(range(6))

COLOR_CHARS = "WHITE GREEN ORANGE YELLOW RED BLUE".split()
SURFACE_CHARS = "TOP LEFT BACK RIGHT FRONT UNDER".split()


class Cube:

    def __init__(self):
        self.state = self._get_initial_cube_arr()

    @property
    def is_solved(self):
        return (self.state == self._get_initial_cube_arr()).all()

    @property
    def is_home_pos(self):
        """HOME POSITION
            TOP  : WHITE
            LEFT : GREEN
            BACK : ORANGE
            RIGHT: YELLOW
            FRONT: RED
            UNDER: BLUE
        """
        cur_w, cur_r = self.current_wr_pos
        home_w, home_r = self.wr_pos_when_home_pos

        return (cur_w == home_w) and (cur_r == home_r)

    @property
    def wr_pos_when_home_pos(self):
        """index of white and red faces
        when the cube in the home position
        """
        return TOP, FRONT

    @property
    def current_wr_pos(self):
        """Return index of white and red faces."""
        center_blocks = self.state[:, 1, 1]
        w = np.argwhere(center_blocks == WHITE)[0, 0]
        r = np.argwhere(center_blocks == RED)[0, 0]

        return w, r

    def __eq__(self, other):
        return (self.state == other.state).all()

    def __ne__(self, other):
        return ~self.__eq__(other)

    def initialize_cube(self):
        self.state = self._get_initial_cube_arr()

    def _get_initial_cube_arr(self):
        return np.array([np.full((3, 3), i) for i in COLORS])

    def copy(self):
        copied = Cube()
        copied.state = self.state.copy()
        return copied
