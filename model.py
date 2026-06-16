# model.py

import torch
import torch.nn as nn


class Attention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.score = nn.Linear(hidden_size, 1)

    def forward(self, x):
        weights = torch.softmax(self.score(x), dim=1)
        return torch.sum(weights * x, dim=1)


class Model(nn.Module):
    def __init__(self, config):
        super().__init__()

        in_channels = config["dataset"]["input_channels"]
        num_classes = config["dataset"]["num_classes"]

        self.conv1 = nn.Conv1d(in_channels, 32, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(kernel_size=2)

        self.conv2 = nn.Conv1d(32, 64, kernel_size=7, padding=3)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(kernel_size=2)

        self.conv3 = nn.Conv1d(64, 64, kernel_size=7, padding=3)
        self.bn3 = nn.BatchNorm1d(64)

        self.conv4 = nn.Conv1d(64, 128, kernel_size=7, padding=3)
        self.bn4 = nn.BatchNorm1d(128)
        self.pool4 = nn.MaxPool1d(kernel_size=2)

        self.conv5 = nn.Conv1d(128, 128, kernel_size=7, padding=3)
        self.bn5 = nn.BatchNorm1d(128)

        self.relu = nn.ReLU()

        self.bigru = nn.GRU(
            input_size=128,
            hidden_size=64,
            batch_first=True,
            bidirectional=True,
        )

        self.attention = Attention(hidden_size=128)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)
        x = self.pool4(x)

        x = self.conv5(x)
        x = self.bn5(x)
        x = self.relu(x)

        x = x.transpose(1, 2)
        x, _ = self.bigru(x)
        x = self.attention(x)
        x = self.dropout(x)
        x = self.fc(x)

        return x
