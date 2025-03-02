from typing import Callable

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import Adam

from models.base import MnistClassifierInterface


class CNNMnistClassifier(MnistClassifierInterface):

    def __init__(self):
        self.model = nn.Sequential(
            nn.Conv2d(1, 32, 3),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(start_dim=1),
            nn.Linear(64 * 5 * 5, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )
        # Specific optimizer and criterion types are defined for each model,
        # allowing the selection of the most suitable options. This also
        # enables passing essential arguments (such as the learning rate)
        # directly from the training loop.
        self.optimizer = Adam
        self.criterion = nn.CrossEntropyLoss

    def train(
        self, data_loader: DataLoader, train_loop: Callable | None = None
    ) -> None:
        train_loop(self.model, self.optimizer, self.criterion, data_loader)

    def predict(self, data: np.ndarray | torch.Tensor) -> np.ndarray:

        if isinstance(data, np.ndarray):
            data = torch.tensor(data)

        with torch.no_grad():
            output = self.model(data)

        return torch.argmax(output, dim=1).numpy()
