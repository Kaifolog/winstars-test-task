from typing import Callable

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.ensemble import RandomForestClassifier

from models.base import MnistClassifierInterface


class RFMnistClassifier(MnistClassifierInterface):

    def __init__(self):
        self.model = RandomForestClassifier(
            max_depth=50, n_estimators=100, random_state=42
        )

    def train(
        self, data_loader: DataLoader, train_loop: Callable | None = None
    ) -> None:

        X, y = [], []

        for images, labels in data_loader:
            X.append(images.view(images.shape[0], -1).numpy())
            y.append(labels.numpy())

        X_train, y_train = np.vstack(X), np.hstack(y)

        self.model.fit(X_train, y_train)

    def predict(self, data: np.ndarray | torch.Tensor) -> np.ndarray:

        if isinstance(data, torch.Tensor):
            data = data.numpy()
        data = data.reshape(data.shape[0], -1)

        return self.model.predict(data)
