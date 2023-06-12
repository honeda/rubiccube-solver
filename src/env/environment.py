from src.env.cube import Cube

ACTIONS = "F F_ L L_ R R_ U U_ D D_ B B_".split()


class Environment:

    def __init__(self):
        self.cube = Cube()
        self.last_scramble_actions = None

    @property
    def actions(self):
        return ACTIONS

    @property
    def reverse_actions(self):
        return {i: f"{i}_" if not i.endswith("_") else i[0] for i in self.actions}

    @property
    def states(self):
        # return self.cube.state
        return " ".join(self.cube.state.ravel().astype(str))

    def set_game_start_position(self, actions: list):
        """Scramble the cube from origin.

        Args:
            actions (list): action list. ex) ["F", "D", "L"]
        """
        self.last_scramble_actions = actions
        self.reset_to_origin()
        for i in actions:
            self.cube.step(i)

    def get_unscramble_actions(self, scramble_actions: list):
        """Return unscramble steps.

        Args:
            scramble_actions (list):
        Return:
            list
        """
        r_act = self.reverse_actions
        return [r_act[i] for i in scramble_actions[::-1]]

    def reset(self):
        """reset to game start"""
        self.set_game_start_position(self.last_scramble_actions)

    def reset_to_origin(self):
        self.cube.reset()

    def step(self, action: int or str):
        """
        Args:
            action (int or str): action
        Returns:
            np.ndarray: state
            int : reward
            bool: is unscrambled
        """
        if type(action) == int:
            self.cube.step(ACTIONS[action])
        else:
            self.cube.step(action)

        return (
            # HACK: もっといいstateのコピー、保存、比較方法
            " ".join(self.cube.state.ravel().astype(str)),
            int(self.cube.is_origin()),
            self.cube.is_origin()
        )

    def is_same_state(self, a, b):
        """
        Args:
            a (np.ndarray): Cube.state
            b (np.ndarray): Cube.state

        Returns:
            bool
        """
        return (a == b).all()
