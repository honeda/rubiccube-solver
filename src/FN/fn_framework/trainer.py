import re
from collections import deque

from src.FN.fn_framework import Experience
from src.env.environment import Environment


class Trainer:

    def __init__(self, buffer_size=1024, batch_size=32,
                 gamma=0.9, report_interval=10, log_dir="."):
        self.buffer_size = buffer_size
        self. batch_size = batch_size
        self.gamma = gamma
        self.report_interval = report_interval
        self.experiences = deque(maxlen=buffer_size)
        self.training = False
        self.training_count = 0
        self.reward_log = []

    @property
    def trainer_name(self):
        class_name = self.__class__.__name__
        snaked = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", class_name)
        snaked = re.sub("([a-z0-9])([A-Z])", r"\1_\2", snaked).lower()
        snaked = snaked.replace("_trainer", "")

        return snaked

    def train_loop(self, env: Environment, agent, episode=200, initial_count=-1,
                   observe_interval=0):
        self.experiences = deque(maxlen=self.buffer_size)
        self.training = False
        self.training_count = 0
        self.reward_log = []
        frames = []

        for i in range(episode):
            s = env.reset_to_gamestart()
            done = False
            step_count = 0
            self.episode_begin(i, agent)

            while not done:
                if (
                    self.training
                    and observe_interval > 0
                    and (self.training_count == 1
                         or self.training_count % observe_interval == 0)
                ):
                    frames.append(s)

                a = agent.policy(s)
                n_state, reward, done = env.step(a)
                e = Experience(s, a, reward, n_state, done)
                self.experiences.append(e)

                if (not self.training) and (len(self.experiences) == self.buffer_size):
                    self.begin_train(i, agent)
                    self.training = True

                self.step(i, step_count, agent, e)

                s = n_state
                step_count += 1

            self.episode_end(i, step_count, agent)

            if (not self.training) and (initial_count > 0) and (i >= initial_count):
                self.begin_train(i, agent)
                self.training = True

            if self.training:
                if len(frames) > 0:
                    frames = []
                self.training_count += 1

    def episode_begin(self, episode, agent):
        raise NotImplementedError

    def begin_train(self, episode, agent):
        raise NotImplementedError

    def step(self, episode, step_count, agent, exprience):
        raise NotImplementedError

    def episode_end(self, episode, step_count, agent):
        raise NotImplementedError

    def is_event(self, count, interval):
        return (count != 0) and (count % interval == 0)

    def get_recent(self, count):
        # HACK: これじゃだめ？検証
        # return self.experiences[-count:]
        recent = range(len(self.experiences) - count, len(self.experiences))
        return [self.experiences[i] for i in recent]
