# src/model.py
import torch.nn as nn
from .config import TrainingConfig  # config.py에서 TrainingConfig 임포트

class ComplexCNN(nn.Module):
    def __init__(self, config: TrainingConfig):
        super(ComplexCNN, self).__init__()
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(in_channels=config.input_channels, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.flattened_dim = 64 * 7 * 7
        self.fc_block = nn.Sequential(
            nn.Linear(self.flattened_dim, 1024),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(1024, config.num_classes)
        )

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = x.view(-1, self.flattened_dim)
        logits = self.fc_block(x)
        return logits