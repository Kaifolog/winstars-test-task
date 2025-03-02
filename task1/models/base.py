from abc import ABC, abstractmethod
from typing import Callable

import torch
from torch.utils.data import DataLoader
import numpy as np


class MnistClassifierInterface(ABC):
    """
    An interface for training and predicting on the MNIST dataset.
    """

    @abstractmethod
    def train(
        self, data_loader: DataLoader, train_loop: Callable | None = None
    ) -> None:
        """
        Train the classifier using the provided data loader.

        Args:
            data_loader (torch.utils.data.DataLoader): The data loader providing training samples.
            train_loop (Callable | None, optional): A custom training loop function.
        """
        pass

    @abstractmethod
    def predict(self, data: np.ndarray | torch.Tensor) -> np.ndarray:
        """
        Predict the class labels for the given input data.

        Args:
            data (np.ndarray | torch.Tensor): The input data, either as a NumPy array
                or a PyTorch tensor.

        Returns:
            np.ndarray: The predicted class labels as a NumPy array.
        """
        pass
