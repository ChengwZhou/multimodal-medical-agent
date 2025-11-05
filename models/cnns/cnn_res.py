import torch
import torch.nn as nn
import torch.nn.functional as F

class SingleModalCNN(nn.Module):
    def __init__(self, in_len=100):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),           # 100 -> 50
            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)    # -> [32, 1]
        )

    def forward(self, x):  # x: [B, 1, 100]
        x = self.conv(x)
        return x.view(x.size(0), -1)  # [B, 32]

class CNN1DResidual(nn.Module):
    def __init__(self, num_modal=14, in_len=100, num_classes=10):
        super().__init__()
        self.modal_nets = nn.ModuleList(
            [SingleModalCNN(in_len) for _ in range(num_modal)]
        )
        self.fc = nn.Sequential(
            nn.Linear(32 * num_modal, 128),
            nn.ReLU(),
            nn.Dropout(0.6),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):  # x: [B, 14, 100]
        feats = []
        for i, net in enumerate(self.modal_nets):
            # if i in [0,1,8,9,10]:
            xi = x[:, i:i+1, :]          # [B,1,100]
            feats.append(net(xi))        # [B,32]
        feats = torch.cat(feats, dim=1)  # [B, 32*14]
        out = self.fc(feats)
        return out

# 例子
if __name__ == "__main__":
    model = CNN1DResidual(num_modal=12, in_len=100, num_classes=6)
    x = torch.randn(8, 12, 100)
    y = model(x)
    print(model)
    print(y.shape)  # [8, 6]
