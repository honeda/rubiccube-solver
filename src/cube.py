import numpy as np

N = 0
W = 1
G = 2
B = 3
Y = 4
O = 5
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
        self.reset()

        self.move_symbols = "F F_ L L_ R R_ U U_ D D_ B B_".split()

    def reset(self):
        self.cube = self.get_initial_cube()

    def get_initial_cube(self):
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

    def play(self, moves):
        """
        Args:
            moves (list): move symbols
        """
        for i in moves:
            eval(f"self.{i}()")

    def get_move_symbols(self):
        return self.move_symbols

    def get_reverse_moves(self, moves):
        """
        Args:
            moves (list): move symbols
        Return:
            list
        """
        ptn = {"F": "F_", "L": "L_", "R": "R_", "U": "U_", "D": "D_", "B": "B_", "F_": "F",
               "L_": "L", "R_": "R", "U_": "U", "D_": "D", "B_": "B"}
        lst = []
        for i in moves[::-1]:
            lst.append(ptn[i])
        return lst

    def F(self):
        self.cube[0,:,:] = np.rot90(self.cube[0,:,:], k=3)

    def F_(self):
        self.cube[0,:,:] = np.rot90(self.cube[0,:,:], k=1)

    def L(self):
        self.cube[:,:,0] = np.rot90(self.cube[:,:,0], k=3)

    def L_(self):
        self.cube[:,:,0] = np.rot90(self.cube[:,:,0], k=1)

    def R(self):
        self.cube[:,:,2] = np.rot90(self.cube[:,:,2], k=1)

    def R_(self):
        self.cube[:,:,2] = np.rot90(self.cube[:,:,2], k=3)

    def U(self):
        self.cube[:,0,:] = np.rot90(self.cube[:,0,:], k=3)

    def U_(self):
        self.cube[:,0,:] = np.rot90(self.cube[:,0,:], k=1)

    def D(self):
        self.cube[:,2,:] = np.rot90(self.cube[:,2,:], k=3)

    def D_(self):
        self.cube[:,2,:] = np.rot90(self.cube[:,2,:], k=1)

    def B(self):
        self.cube[2,:,:] = np.rot90(self.cube[2,:,:], k=1)

    def B_(self):
        self.cube[2,:,:] = np.rot90(self.cube[2,:,:], k=3)

    def show_cube(self):
        lst = []
        for i in self.cube.ravel():
            lst.append(COLORS[i])
        return np.array(lst).reshape(3, 3, 3)

    def check_origin(self):
        return (self.cube == self.get_initial_cube()).all()