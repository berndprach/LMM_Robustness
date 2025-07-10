from functools import partial

from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import transforms as tfs

from .split_deterministically import split_deterministically

CHANNEL_MEANS = (0.4914, 0.4822, 0.4465)
CLASS_NAMES = ["plane", "car", "bird", "cat", "deer",
               "dog", "frog", "horse", "ship", "truck"]
CLASS_ID = {name: i for i, name in enumerate(CLASS_NAMES)}
TRANSFORM = {
    "to_tensor": tfs.ToTensor(),
    None: None
}

get_cifar = partial(CIFAR10, root="data", download=True)


def get_data(sizes=(45_000, 5_000), transform=None):
    train_val_ds = get_cifar(train=True, transform=transform)
    train_ds, val_ds = split_deterministically(train_val_ds, sizes)
    return train_ds, val_ds


def get_data_loader(sizes=(45_000, 5_000), batch_size=256, tf: str = None):
    train_val_ds = get_cifar(train=True, transform=TRANSFORM[tf])
    train_ds, val_ds = split_deterministically(train_val_ds, sizes)

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    return train_dl, val_dl


