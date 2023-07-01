import json
from concurrent.futures import ThreadPoolExecutor
from logging import config, getLogger

import numpy as np
import torch

from src.env.action import ACTION_NUMS
from src.neural_network.data import QTable

LOGGER_CONF_FILE = "config/logger.json"
GAMMA = 0.9  # モンテカルロで使ったgammaの値. 手数ペナルティの役割


class DataLoader:

    def __init__(self, q_file, batch_size, device, shuffle=False):
        """
        Args:
            q_file (str): Path to Q file. Defaults to None.
            batch_size (int): Batch size.
            device (torch.device): "cpu" or "gpu".
            shuffle (bool, optional): Defaults to False.
        """
        self._setup_logger()
        self.batch_size = batch_size
        self.device = device
        self.shuffle = shuffle

        if q_file:
            self.load(q_file)
        else:
            pass

        if device == torch.device("cpu"):
            pin_memory = False
        else:
            pin_memory = True

        self.torch_fetures = torch.empty(
            (batch_size, 6, 3, 3), dtype=torch.float32, pin_memory=pin_memory
        )
        self.torch_label = torch.empty(
            (batch_size, len(ACTION_NUMS)), dtype=torch.float32, pin_memory=pin_memory
        )
        self.torch_value = torch.empty(
            (batch_size, 1), dtype=torch.float32, pin_memory=pin_memory
        )

        self.features = self.torch_fetures.numpy()
        self.label = self.torch_label.numpy()
        self.value = self.torch_value.numpy().ravel()

        self.i = 0
        self.executor = ThreadPoolExecutor(max_workers=1)

    def load(self, q_file, remove_low_value=True):
        """
        Args:
            q_file (str): Q file path
            remove_low_value (bool, optional) If True, remove low value data.
                Defaults to True.
        """
        # states, q_values = [], []
        # for s, v in Q.items():
        #     states.append(s)
        #     q_values.append(v)

        # states = np.array(states)
        # q_values = np.array(q_values)
        q_table = np.fromfile(q_file, dtype=QTable)

        if remove_low_value:
            th = GAMMA ** 30  # 30手かかる局面の価値より小さいデータを除去する.
            mask = (np.max(q_table["action"], axis=1) > th)
            q_table = q_table[mask]
            self.logger.info(f"Remove {np.sum(~mask)} low value data.")

        self.q_table = q_table

    def mini_batch(self, q_tbl):
        self.features.fill(0)
        for i, d in enumerate(q_tbl):
            self.features[i] = d["state"].reshape(6, 3, 3)
            self.label[i] = self.make_label(d["action"])
            self.value[i] = self.make_value(d["action"])

        if self.device.type == "cpu":
            return (
                self.torch_fetures.clone(),
                self.torch_label.clone(),
                self.torch_value.clone()
            )
        else:
            return (
                self.torch_fetures.to(self.device),
                self.torch_label.to(self.device),
                self.torch_value.to(self.device)
            )

    def sample(self):
        return self.mini_batch(np.random.choice(self.q_table, self.batch_size, replace=False))

    def pre_fetch(self):
        q_tbl = self.q_table[self.i: self.i + self.batch_size]
        self.i += 1
        if len(q_tbl) < self.batch_size:
            return

        self.f = self.executor.submit(self.mini_batch, q_tbl)

    def __len__(self):
        return len(self.q_table)

    def __iter__(self):
        self.i = 0
        if self.shuffle:
            np.random.shuffle(self.q_table)
        self.pre_fetch()

        return self

    def __next__(self):
        if self.i > len(self.q_table):
            raise StopIteration()

        result = self.f.result()
        self.pre_fetch()

        return result

    def make_value(self, values):
        """
        Args:
            values (list): Values of Q-table.
        Return:
            float
        """
        return np.max(values)

    def make_label(self, values):
        """
        Args:
            values (list): Values of Q-table.
        Return:
            np.ndarray: Shape is (len(ACTION_NUMS),)
        """
        # 最短手数が2手の場合、価値は最大で gamma**(2-1) = 0.9**1 = 0.9
        # 最短手数が3手の場合、価値は最大で gamma**(3-1) = 0.81
        # 3手かかる場合、価値が0.81以上になることはない.
        # つまり 0.81 < x <= 0.9 のアクションを選べば2手で終わるということになるので
        # この範囲のアクションがいくつかある場合はすべて2手で終わるということになる.

        bins = np.array([GAMMA ** i for i in range(50)])
        idx = np.digitize(np.max(values), bins)

        label = ((values <= bins[idx] + 1e6) & (values > bins[idx + 1])).astype(float)
        if sum(label) == 0:
            msg = "Label preprocessing error."
            self.logger.error(msg)
            raise Exception(msg)

        return label

    def _setup_logger(self):
        log_conf = json.load(open(LOGGER_CONF_FILE))
        log_conf["handlers"]["fileHandler"]["filename"] = f"log/{self.__class__.__name__}.log"

        config.dictConfig(log_conf)
        self.logger = getLogger("simpleLogger")
