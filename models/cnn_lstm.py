import torch
import torch.nn as nn

class CNNLSTM(nn.Module):
    def __init__(self, input_dim=12, hidden_dim=64, lstm_layers=1, num_classes=13):
        """
        input_dim: Number of features per time step
        hidden_dim: LSTM hidden size
        num_classes: Number of classification categories (mHealth has 13 activity classes)
        """
        super().__init__()
        self.conv1 = nn.Conv1d(input_dim, 32, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm1d(32)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm1d(64)
        self.relu2 = nn.ReLU()
        self.pool  = nn.MaxPool1d(2)

        # LSTM 捕捉长序列依赖
        self.lstm = nn.LSTM(input_size=64, hidden_size=hidden_dim,
                            num_layers=lstm_layers, batch_first=True)

        self.fc1 = nn.Linear(hidden_dim, 128)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        # x: (batch, time_steps, features)                # (batch, features, time_steps)
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = x.transpose(1,2)                  # (batch, time, channels)

        out, _ = self.lstm(x)                 # (batch, time, hidden_dim)
        out = out[:, -1, :]                   # last time step
        out = self.fc2(self.relu3(self.fc1(out)))
        return out


if __name__ == "__main__":
    model = CNNLSTM(num_classes=6)
    x = torch.randn(8, 100, 12)
    y = model(x)
    print(model)
    print(y.shape)  # [8, 6]
