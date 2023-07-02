import torch
import torch.nn as nn

from src.env.action import ACTION_NUMS

N_CHANNEL = 1


class ResNetBlock(nn.Module):
    def __init__(self, channels):
        super(ResNetBlock, self).__init__()

        self.relu = nn.ReLU()

        # 畳み込み層の後ろがBatchNormalization層の場合はbiasは省略.
        # ブロックの位置が重要なためダウンサンプリング(MaxPool)は行わない.
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        out = self.relu(out + x)

        return out


class Model(nn.Module):
    """
    bn: True
    dropout: False
    resnet: True

    * Policyの出力層について
     今回のモデルでは正解ラベルが2つ以上ある(最短手数のアクションが2つ以上ある)
     場合があるのでSoftmax + Crossentropy ではなく、
     Sigmoid + BinaryCrossentropy を使う.(多ラベル分類)
    """
    def __init__(self, channels=64, blocks=10):
        super(Model, self).__init__()

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        self.conv1 = nn.Conv2d(in_channels=N_CHANNEL, out_channels=channels,
                               kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)

        self.blocks = nn.Sequential(*[ResNetBlock(channels) for _ in range(blocks)])

        # policy
        self.policy_conv = nn.Conv2d(in_channels=channels, out_channels=len(ACTION_NUMS),
                                     kernel_size=1, bias=False)
        self.policy_bn = nn.BatchNorm2d(len(ACTION_NUMS))
        self.policy_fc1 = nn.Linear(len(ACTION_NUMS) * 6 * 9, 32)
        self.policy_fc2 = nn.Linear(32, len(ACTION_NUMS))

        # value
        self.value_conv = nn.Conv2d(in_channels=channels, out_channels=N_CHANNEL,
                                    kernel_size=1, bias=False)
        self.value_bn = nn.BatchNorm2d(N_CHANNEL)
        self.value_fc1 = nn.Linear(N_CHANNEL * 6 * 9, 32)
        self.value_fc2 = nn.Linear(32, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        # resnet blocks
        x = self.blocks(x)

        policy = self.relu(self.policy_bn(self.policy_conv(x)))
        policy = self.relu(self.policy_fc1(torch.flatten(policy, 1)))
        policy = self.sigmoid(self.policy_fc2(policy))

        value = self.relu(self.value_bn(self.value_conv(x)))
        value = self.relu(self.value_fc1(torch.flatten(value, 1)))
        value = self.sigmoid(self.value_fc2(value))

        return policy, value
