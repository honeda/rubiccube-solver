import torch.nn as nn

from src.env.action import ACTION_NUMS, SURFACE_CHARS

N_CHANNEL = len(SURFACE_CHARS)


class Model(nn.Module):
    """
    bn: False
    dropout: False
    resnet: False

    * Policyの出力層について
     今回のモデルでは正解ラベルが2つ以上ある(最短手数のアクションが2つ以上ある)
     場合があるのでSoftmax + Crossentropy ではなく、
     Sigmoid + BinaryCrossentropy を使う.(多ラベル分類)
    """
    def __init__(self):
        super(Model, self).__init__()

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        # ブロックの位置が重要なためダウンサンプリング(MaxPool)は行わない.
        self.conv1 = nn.Conv2d(in_channels=N_CHANNEL, out_channels=N_CHANNEL * 2,
                               kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=N_CHANNEL * 2, out_channels=N_CHANNEL * 2,
                               kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=N_CHANNEL * 2, out_channels=N_CHANNEL,
                               kernel_size=3, padding=1)

        self.flatten = nn.Flatten()

        # policy
        self.policy_fn1 = nn.Linear(N_CHANNEL * 3 * 3, 28)
        self.policy_fn2 = nn.Linear(28, len(ACTION_NUMS))

        # value
        self.value_fn1 = nn.Linear(N_CHANNEL * 3 * 3, 28)
        self.value_fn1 = nn.Linear(28, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.relu(x)

        x = self.flatten(x, 1)

        policy = self.policy_fn1(x)
        policy = self.relu(policy)
        policy = self.policy_fn2(policy)
        policy = self.sigmoid(policy)

        value = self.value_fn1(x)
        value = self.relu(value)
        value = self.value_fn2(value)
        value = self.sigmoid(value)

        return policy, value
