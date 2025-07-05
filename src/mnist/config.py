# src/config.py
import torch
from dataclasses import dataclass

@dataclass
class TrainingConfig:
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    input_channels: int = 1
    num_classes: int = 10
    num_epochs: int = 3
    batch_size: int = 128
    learning_rate: float = 0.01
    weight_decay: float = 1e-4  # L2 Regularization을 위한 가중치 감쇠
    dropout_rate: float = 0.5