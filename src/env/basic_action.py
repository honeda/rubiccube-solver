import numpy as np

from src.env.cube import (
    Cube,
    TOP, LEFT, BACK, RIGHT, FRONT, UNDER,
)


def action_base(cube: Cube, cb):
    cube.state = cb(cube.state)


def X(cube: Cube):

    def cb(c):
        # rotation
        # c[TOP] = c[TOP]
        c[LEFT] = np.rot90(c[LEFT], k=1)
        c[BACK] = np.flip(c[BACK])
        c[RIGHT] = np.rot90(c[RIGHT], k=3)
        # c[FRONT] = c[FRONT]
        c[UNDER] = np.flip(c[UNDER])
        # move
        c = c[[FRONT, LEFT, TOP, RIGHT, UNDER, BACK]]

        return c

    return action_base(cube, cb)


def X_(cube: Cube):
    # For speed. NO use X(X(X(cube)))

    def cb(c):
        # rotation
        # c[TOP] = c[TOP]
        c[LEFT] = np.rot90(c[LEFT], k=3)
        # c[BACK] = c[BACK]
        c[RIGHT] = np.rot90(c[RIGHT], k=1)
        c[FRONT] = np.flip(c[FRONT])
        c[UNDER] = np.flip(c[UNDER])
        # move
        c = c[[BACK, LEFT, UNDER, RIGHT, TOP, FRONT]]

        return c

    return action_base(cube, cb)


def Y(cube: Cube):

    def cb(c):
        # rotation
        lst = [TOP, LEFT, BACK, RIGHT, FRONT]
        c[lst] = np.rot90(c[lst], k=3, axes=(1, 2))
        c[UNDER] = np.rot90(c[UNDER], k=1)
        # move
        c = c[[TOP, FRONT, LEFT, BACK, RIGHT, UNDER]]

        return c

    return action_base(cube, cb)


def Y_(cube: Cube):
    # For speed.  NO use Y(Y(Y(cube)))

    def cb(c):
        # rotation
        lst = [TOP, LEFT, BACK, RIGHT, FRONT]
        c[lst] = np.rot90(c[lst], k=1, axes=(1, 2))
        c[UNDER] = np.rot90(c[UNDER], k=3)
        # move
        c = c[[TOP, BACK, RIGHT, FRONT, LEFT, UNDER]]

        return c

    return action_base(cube, cb)


def F(cube: Cube):

    def cb(c):
        # front
        c[FRONT] = np.rot90(c[FRONT], k=3)
        # left, under, right, top
        top = c[LEFT][-1].copy()
        c[LEFT][-1] = c[UNDER][-1]
        c[UNDER][-1] = c[RIGHT][-1]
        c[RIGHT][-1] = c[TOP][-1]
        c[TOP][-1] = top

        return c

    return action_base(cube, cb)


def F_(cube: Cube):
    # For speed. NO use F(F(F(cube)))

    def cb(c):
        # front
        c[FRONT] = np.rot90(c[FRONT], k=1)
        # left, under, right, top
        under = c[LEFT][-1].copy()
        c[LEFT][-1] = c[TOP][-1]
        c[TOP][-1] = c[RIGHT][-1]
        c[RIGHT][-1] = c[UNDER][-1]
        c[UNDER][-1] = under

        return c

    return action_base(cube, cb)


def Z(cube: Cube):
    actions = [Y, X_, Y_]
    for a in actions:
        a(cube)


def Z_(cube: Cube):
    actions = [Y, X, Y_]
    for a in actions:
        a(cube)


def R(cube: Cube):
    actions = [Y, F, Y_]
    for a in actions:
        a(cube)


def R_(cube: Cube):
    actions = [Y, F_, Y_]
    for a in actions:
        a(cube)
