import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN1DResidual(nn.Module):
    def __init__(self, num_modal=14, in_len=100, num_classes=10):
        super(CNN1DResidual, self).__init__()
        self.n_chan = num_modal
        self.n_classes = num_classes

        # Convolutional Layers
        self.conv1 = nn.Conv1d(self.n_chan, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1)
        self.drop = nn.Dropout(p=0.6)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

        # Fully connected layers
        self.lin3 = nn.Linear(3200, 100)
        self.lin4 = nn.Linear(100, self.n_classes)

    def forward(self, x):
        batch_size = x.size(0)

        a = torch.relu(self.conv1(x))
        # print(a.shape)
        a = torch.relu(self.conv2(a))
        # print(a.shape)
        a = self.drop(a)
        a = self.pool(a)
        # print(a.shape)
        a = a.view(batch_size, -1)
        a = torch.relu(self.lin3(a))
        a = torch.relu(self.lin4(a))
        return a


if __name__ == "__main__":
    model = CNN1DResidual(num_modal=14, in_len=100, num_classes=6)
    x = torch.randn(8, 14, 100)
    y = model(x)
    print(model)
    print(y.shape)  # [8, 6]
