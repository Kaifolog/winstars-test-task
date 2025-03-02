from torchvision import datasets, transforms
import torch
from torch.utils.data import DataLoader

transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(
            (0.1307,), (0.3081,)  # global mean and std of dataset pixels
        ),
    ]
)

train_set = datasets.MNIST(
    "./datasets/", train=True, download=True, transform=transform
)
test_set = datasets.MNIST(
    "./datasets/", train=False, download=True, transform=transform
)


def get_dataloader(dataset: datasets, batch_size: int, shuffle: bool):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
