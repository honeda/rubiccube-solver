import numpy as np
import torch
from torch.utils.data import Dataset

from src.neural_network.data import QTable
from src.neural_network.data.features import make_multi_label, make_value


class MultiLabelDataset(Dataset):
    """
    Input : (6, 3, 3)
    Output: (12,), (1,)
    """

    def __init__(self, path):
        """
        Args:
            path (str): Path of structured-array file.
            device (torch.device):
        """
        super().__init__()

        self.data = np.fromfile(path, dtype=QTable)

        self.states = torch.Tensor(self.data["state"])
        self.labels = torch.Tensor(
            np.array([make_multi_label(i) for i in self.data["action"]])
        )
        self.values = torch.Tensor(
            np.array([make_value(i) for i in self.data["action"]])
        )
        self.length = len(self.data)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        s = self.states[index].reshape(6, 3, 3)
        label = self.labels[index]
        value = self.values[index].reshape(1,)

        return s, label, value


class MultiLabelDatasetOneChannel(Dataset):
    """
    Input : (1, 6, 9)
    Output: (12,), (1,)
    """

    def __init__(self, path):
        """
        Args:
            path (str): Path of structured-array file.
            device (torch.device):
        """
        super().__init__()

        self.data = np.fromfile(path, dtype=QTable)

        self.states = torch.Tensor(self.data["state"])
        self.labels = torch.Tensor(
            np.array([make_multi_label(i) for i in self.data["action"]])
        )
        self.values = torch.Tensor(
            np.array([make_value(i) for i in self.data["action"]])
        )
        self.length = len(self.data)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        s = self.states[index].reshape(1, 6, 9)
        label = self.labels[index]
        value = self.values[index].reshape(1,)

        return s, label, value
