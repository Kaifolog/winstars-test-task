from typing import Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from tqdm import tqdm

from models.rf import RFMnistClassifier
from models.nn import FFNMnistClassifier
from models.cnn import CNNMnistClassifier


class MnistClassifier:
    """
    A unified interface for MNIST image classification using different models.

    This class provides a common API for training and prediction with different models
    (Random Forest, Feed-Forward Neural Network, or Convolutional Neural Network).
    """

    def __init__(self, label: str):
        """
        Initialize the MNIST classifier with the specified model type.

        Args:
            label (str): Type of model to use. Must be one of:
                - "rf": Random Forest classifier
                - "nn": Feed-Forward Neural Network
                - "cnn": Convolutional Neural Network

        Raises:
            ValueError: If the label is not one of the supported model types.
        """
        match label:
            case "rf":
                self._classifier = RFMnistClassifier()
            case "nn":
                self._classifier = FFNMnistClassifier()
            case "cnn":
                self._classifier = CNNMnistClassifier()
            case _:
                raise ValueError("Model label should be rf, nn or cnn.")

    def _train_loop(
        self,
        model: nn.Module | nn.Sequential,
        optimizer: type[torch.optim.Optimizer],
        criterion: type[nn.Module],
        data_loader: DataLoader,
    ) -> None:
        """
        Internal training loop callback for neural network models.

        Performs the epoch-based training process for PyTorch models.

        Args:
            model: PyTorch model to train.
            optimizer: PyTorch optimizer class to use for training.
            criterion: PyTorch loss function class to use for training.
            data_loader: DataLoader containing the training data.
        """

        n_epochs = self._loop_args["epochs"]
        optimizer = optimizer(model.parameters(), lr=self._loop_args["lr"])
        criterion = criterion()

        model.train()
        for i in range(n_epochs):
            epoch_loss = 0.0

            for images, labels in tqdm(
                data_loader, desc=f"Epoch {i+1}/{n_epochs}", unit="batch"
            ):
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            print(f"Epoch {i+1}/{n_epochs}, Loss: {epoch_loss / len(data_loader)}")

    def train(self, data_loader: DataLoader, epochs: int = 10, lr: float = 0.1) -> None:
        """
        Train the classifier on the provided data.

        Args:
            data_loader (DataLoader): DataLoader containing training data.
            epochs (int, optional): Number of training epochs. Defaults to 10.
            lr (float, optional): Learning rate for optimization. Defaults to 0.1.

        Returns:
            None: The model is trained in-place.
        """
        self._loop_args = {"epochs": epochs, "lr": lr}
        return self._classifier.train(data_loader, self._train_loop)

    def predict(
        self, data: DataLoader | np.ndarray, metrics: bool = False
    ) -> np.ndarray | Tuple[np.ndarray, dict]:
        """
        Generate predictions using the trained classifier.

        This method handles different input formats (DataLoader or numpy arrays)
        and can optionally compute performance metrics when labels are available.

        Args:
            data (DataLoader | np.ndarray): Input data for prediction.
                - If DataLoader: Should yield batches of (images, labels) or just images.
                - If numpy array: Normalized features with shape (picturesnumber, 1, 28, 28).
            metrics (bool, optional): Whether to compute and return performance metrics.
                Requires labels to be available. Defaults to False.

        Returns:
            np.ndarray | Tuple[np.ndarray, dict]:
                - If metrics=False: Array of predicted class labels.
                - If metrics=True: Tuple containing predictions array and
                  a dictionary of metrics (accuracy, F1 score, confusion matrix).
        """

        y_true, y_pred = [], []
        if isinstance(data, DataLoader):
            for batch in data:
                if len(batch) == 2:
                    X, labels = batch
                    preds = self._classifier.predict(X)

                    y_true.append(labels.numpy())
                    y_pred.append(preds)
                else:
                    X = batch
                    preds = self._classifier.predict(X)
                    y_pred.append(preds)
        else:
            if data.shape[1:] == (1, 28, 28):
                X = data
                preds = self._classifier.predict(X)
                y_pred.append(preds)
            else:
                raise ValueError(
                    "Shape of np.ndarray should be (picturesnumber, 1, 28, 28)"
                )

        y_pred = np.concatenate(y_pred, axis=0)
        y_true = np.concatenate(y_true, axis=0) if len(y_true) > 0 else None

        if metrics and y_true is not None:

            acc = accuracy_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred, average="macro")
            cm = confusion_matrix(y_true, y_pred)

            return y_pred, {"accuracy": acc, "f1": f1, "confusion_matrix": cm}
        else:
            return y_pred
