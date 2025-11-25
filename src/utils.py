from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import transforms


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


# TODO: Rewrite to just accept config params.
def get_mnist_dataloader(
    data_dir: Path,
    batch_size: int = 32,
    resize_size: int = 32,
):
    transform = transforms.Compose(
        [
            transforms.Resize(resize_size),
            transforms.ToTensor(),
        ]
    )
    train_ds = MNIST(
        root=str(data_dir),
        train=True,
        transform=transform,
        download=True,
    )
    return DataLoader(train_ds, batch_size=batch_size, shuffle=True)
