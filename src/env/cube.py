import numpy as np


COLORS = (
    WHITE,
    ORANGE,
    BLUE,
    RED,
    GREEN,
    YELLOW
) = list(range(6))

SURFACES = (
    TOP,
    LEFT,
    BACK,
    RIGHT,
    FRONT,
    UNDER
) = list(range(6))

COLOR_CHARS = "WHITE ORANGE BLUE RED GREEN YELLOW".split()
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
            LEFT : ORANGE
            BACK : BLUE
            RIGHT: RED
            FRONT: GREEN
            UNDER: YELLOW
        """
        cur_w, cur_g = self.current_wg_pos
        home_w, home_g = self.wg_pos_when_home_pos

        return (cur_w == home_w) and (cur_g == home_g)

    @property
    def wg_pos_when_home_pos(self):
        """index of white and red faces
        when the cube in the home position
        """
        return TOP, FRONT

    @property
    def current_wg_pos(self):
        """Return index of white and red faces."""
        center_blocks = self.state[:, 1, 1]
        w = np.argmax(center_blocks == WHITE)
        g = np.argmax(center_blocks == GREEN)

        return w, g

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
