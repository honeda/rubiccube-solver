from src.env.cube import Cube
from src.env import action as ACTION
from src.utils.cube_util import encode_state


class Environment:

    def __init__(self):
        self.cube = Cube()
        self.last_scramble_actions = None

    @property
    def actions(self):
        return ACTION.ACTION_CHARS

    @property
    def states(self):
        return encode_state(self.cube)

    def set_game_start_position(self, actions: list):
        """Scramble the cube from origin.

        Args:
            actions (list): action list. ex) ["F", "D", "L"] or [0, 1, 4]
        """
        self.last_scramble_actions = actions
        self.reset_to_solved()
        ACTION.steps(self.cube, actions)

    def get_unscramble_actions(self, scramble_actions: list):
        """Return solve steps.

        Args:
            scramble_actions (list):
        Return:
            list
        """
        return ACTION.get_reverse_actions(scramble_actions, return_type="int")

    def reset_to_gamestart(self):
        self.set_game_start_position(self.last_scramble_actions)

    def reset_to_solved(self):
        self.cube.initialize_cube()

    def step(self, action: int or str):
        """
        Args:
            action (int or str): action
        Returns:
            np.ndarray: state
            int : reward
            bool: is unscrambled
        """
        ACTION.step(self.cube, action)

        return (
            encode_state(self.cube),
            int(self.cube.is_solved),
            self.cube.is_solved
        )
