import numpy as np

N = 0
W = 1
G = 2
B = 3
Y = 4
O = 5  # noqa: E741
R = 6
WG = 7
WO = 8
WY = 9
WR = 10
GB = 11
GO = 12
GR = 13
BO = 14
BR = 15
BY = 16
YO = 17
YR = 18
WGO = 19
WOY = 20
WYR = 21
WRG = 22
GBO = 23
GRB = 24
BYO = 25
BRY = 26

COLORS = (
    "N W G B Y O R WG WO WY WR GB GO GR BO BR BY YO "
    "YR WGO WOY WYR WRG GBO GRB BYO BRY".split()
)


class Cube:

    def __init__(self):
        self._state = self.get_initial_state()

    @property
    def state(self):
        return self._state

    def reset(self):
        self._state = self.get_initial_state()

    def get_initial_state(self):
        return np.array([
            [[WGO, WO, WOY],
             [WG, W, WY],
             [WRG, WR, WYR]],
            [[GO, O, YO],
             [G, N, Y],
             [GR, R, YR]],
            [[GBO, BO, BYO],
             [GB, B, BY],
             [GRB, BR, BRY]]
        ])

    def step(self, action):
        if action == "F":
            self._state[0, :, :] = np.rot90(self._state[0, :, :], k=3)

        elif action == "F_":
            self._state[0, :, :] = np.rot90(self._state[0, :, :], k=1)

        elif action == "L":
            self._state[:, :, 0] = np.rot90(self._state[:, :, 0], k=3)

        elif action == "L_":
            self._state[:, :, 0] = np.rot90(self._state[:, :, 0], k=1)

        elif action == "R":
            self._state[:, :, 2] = np.rot90(self._state[:, :, 2], k=1)

        elif action == "R_":
            self._state[:, :, 2] = np.rot90(self._state[:, :, 2], k=3)

        elif action == "U":
            self._state[:, 0, :] = np.rot90(self._state[:, 0, :], k=3)

        elif action == "U_":
            self._state[:, 0, :] = np.rot90(self._state[:, 0, :], k=1)

        elif action == "D":
            self._state[:, 2, :] = np.rot90(self._state[:, 2, :], k=3)

        elif action == "D_":
            self._state[:, 2, :] = np.rot90(self._state[:, 2, :], k=1)

        elif action == "B":
            self._state[2, :, :] = np.rot90(self._state[2, :, :], k=1)

        elif action == "B_":
            self._state[2, :, :] = np.rot90(self._state[2, :, :], k=3)

        else:
            raise Exception

    def is_origin(self):
        return (self._state == self.get_initial_state()).all()

    def show_cube(self):
        lst = []
        for i in self._state.ravel():
            lst.append(COLORS[i])
        return np.array(lst).reshape(3, 3, 3)
