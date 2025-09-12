import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, downsample=None):
        super(ResidualBlock1D, self).__init__()
        padding = kernel_size // 2

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.downsample = downsample  # 用于匹配残差的维度

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class CNN1DResidual(nn.Module):
    def __init__(self, num_classes: int):
        super(CNN1DResidual, self).__init__()

        # 输入 [B, 14, T]
        self.stem = nn.Sequential(
            nn.Conv1d(14, 32, kernel_size=7, stride=2, padding=3, bias=False),  # 降采样一半
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2)  # 再降一半
        )

        # 残差层
        self.layer1 = self._make_layer(32, 64, stride=2)   # [B, 64, T/8]
        self.layer2 = self._make_layer(64, 128, stride=2)  # [B, 128, T/16]
        self.layer3 = self._make_layer(128, 256, stride=2) # [B, 256, T/32]

        # 全局池化
        self.global_pool = nn.AdaptiveAvgPool1d(1)  # [B, 256, 1]

        # 分类头
        self.fc = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def _make_layer(self, in_channels, out_channels, stride=1):
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )
        return nn.Sequential(
            ResidualBlock1D(in_channels, out_channels, stride=stride, downsample=downsample),
            ResidualBlock1D(out_channels, out_channels)
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.global_pool(x)  # [B, 256, 1]
        x = x.view(x.size(0), -1)  # [B, 256]
        x = self.fc(x)            # [B, num_classes]
        return x
