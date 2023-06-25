from src.env import Environment

class Observer:

    def __init__(self, env: Environment):
        self._env = env

    @property
    def action_space(self):
        return self._env.actions

    # @property
    # def observation_space(self):
    #     return self._env.observation_space

    def reset(self):
        return self.transform(self._env.reset_to_gamestart())

    def step(self, action):
        n_state, reward, done = self._env.step(action)
        return self.transform(n_state), reward, done

    def transform(self, state):
        raise NotImplementedError
