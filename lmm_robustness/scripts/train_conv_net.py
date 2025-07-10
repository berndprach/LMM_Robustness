import os
from pathlib import Path

import torch
from torch.optim import SGD
from torch.optim.lr_scheduler import OneCycleLR
from torchvision.transforms import transforms as tfs

from lmm_robustness.architecture import simple_conv_net
from lmm_robustness.architecture.center import CenterImage
from lmm_robustness.data import cifar10

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LR = 0.1
EPOCHS = 24
STATE_DICT_PATH = Path("outputs", "state_dict.pt")
os.makedirs(STATE_DICT_PATH.parent, exist_ok=True)


def get_augmentation(h=32, w=32, crop_size=4):
    crop = tfs.RandomCrop((h, w), padding=crop_size, padding_mode="reflect")
    flip = tfs.RandomHorizontalFlip()
    erase = tfs.RandomErasing(p=1., scale=(1 / 16, 1 / 16), ratio=(1., 1.))
    return tfs.Compose([crop, flip, erase])


class CrossEntropyWithTemperature:
    def __init__(self, temperature=1., **kwargs):
        self.temperature = temperature
        self.cross_entropy = torch.nn.CrossEntropyLoss(**kwargs)

    def __call__(self, score_batch, label_batch):
        score_batch /= self.temperature
        return self.cross_entropy(score_batch, label_batch) * self.temperature


def repeat(iterable, times):
    for _ in range(times):
        for item in iterable:
            yield item


def train_model():
    train_dl, val_dl = cifar10.get_data_loader(tf="to_tensor")
    training_steps = len(train_dl) * EPOCHS

    model = simple_conv_net.get_model()
    model.to(DEVICE)

    loss_function = CrossEntropyWithTemperature(temperature=8)
    optimizer = SGD(model.parameters(), lr=0., momentum=0.9, nesterov=True)
    scheduler = OneCycleLR(optimizer, LR, total_steps=training_steps)
    augment = get_augmentation()
    center = CenterImage(cifar10.CHANNEL_MEANS, DEVICE)

    # Train:
    for i, (x_batch, y_batch) in enumerate(repeat(train_dl, EPOCHS)):
        x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)
        x_batch = center(x_batch)
        x_batch = augment(x_batch)

        predictions = model(x_batch)
        loss = loss_function(predictions, y_batch)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()

        log_str = f"Step {i: 4d} / {training_steps:d}, Loss: {loss.item():.3f}"
        print(log_str, end="\r")

    # Validation:
    val_accuracies = []
    for x_batch, y_batch in val_dl:
        x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)
        x_batch = center(x_batch)
        predictions = model(x_batch)

        batch_accuracies = torch.eq(predictions.argmax(dim=1), y_batch)
        val_accuracies.extend(batch_accuracies.detach().tolist())

    val_accuracy = sum(val_accuracies) / len(val_accuracies)
    print(f"\nValidation Accuracy: {val_accuracy:.1%}")

    # Save model state-dict:
    torch.save(model.state_dict(), STATE_DICT_PATH)
    print(f"\nSaved model state-dict to {STATE_DICT_PATH}.")

    return model


def main():
    train_model()


if __name__ == "__main__":
    main()
