import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN1DModel(nn.Module):
    def __init__(self, num_classes: int):
        super(CNN1DModel, self).__init__()

        self.conv1 = nn.Conv1d(in_channels=14, out_channels=32, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(2)  # [B, 32, 50]

        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(2)  # [B, 64, 25]

        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(128)
        self.pool3 = nn.AdaptiveMaxPool1d(1)  # [B, 128, 1]

        self.fc1 = nn.Linear(128, 64)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        # x shape: [B, 14, 100]
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = x.view(x.size(0), -1)  # [B, 128]
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x
