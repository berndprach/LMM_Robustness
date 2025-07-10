import os
import sys
from functools import partial
from pathlib import Path

import torch
import torchattacks
import yaml
from torch import nn
from torch.linalg import vector_norm

from lmm_robustness.algorithms.my_attack import MyAttack
from lmm_robustness.algorithms.my_rounding_attack import MyRoundingAttack
from lmm_robustness.architecture import simple_conv_net
from lmm_robustness.architecture.center import CenterImage
from lmm_robustness.data import cifar10
from lmm_robustness.data.cifar10 import get_data_loader
from lmm_robustness.scripts.train_conv_net import STATE_DICT_PATH

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUTS = Path("outputs", "boundary_distances")
os.makedirs(OUTPUTS, exist_ok=True)


def load_trained_model():
    model = simple_conv_net.get_model()
    state_dict = torch.load(
        STATE_DICT_PATH, map_location=torch.device('cpu'), weights_only=True
    )
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    return model


def get_accuracy(model, data_loader, attack=lambda x, y: x):
    correct = 0
    total = 0
    with torch.no_grad():
        for x_batch, y_batch in data_loader:
            x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)
            x_batch = attack(x_batch, y_batch)
            scores = model(x_batch)
            predictions = scores.argmax(dim=1)
            total += y_batch.size(0)
            correct += predictions.eq(y_batch).sum().item()
    return correct / total


def get_distances(model, data_loader, attack, max_distance=100.):
    all_distances = []
    for x_batch, y_batch in data_loader:
        x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)
        adversarial_examples = attack(x_batch, y_batch)
        adversarial_predictions = model(adversarial_examples)
        perturbations = adversarial_examples - x_batch
        distances = vector_norm(perturbations, ord=2, dim=(1, 2, 3))
        distances = torch.where(
            adversarial_predictions.argmax(dim=1).eq(y_batch),
            torch.tensor(max_distance, device=DEVICE),
            distances,
        )
        all_distances.extend(distances.tolist())
    return all_distances


def get_loss(model, data_loader, attack=lambda x, y: x):
    total_loss = 0.0
    total_samples = 0
    for x_batch, y_batch in data_loader:
        x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)
        x_batch = attack(x_batch, y_batch)
        scores = model(x_batch)
        loss = torch.nn.functional.cross_entropy(scores, y_batch)
        total_loss += loss.item() * y_batch.size(0)
        total_samples += y_batch.size(0)
    return total_loss / total_samples if total_samples > 0 else 0.0


ATTACKS = {
    "FAB": partial(torchattacks.FAB, norm="L2", eps=10.),
    "MyAttack": partial(MyAttack, eps=0.5),
    "MyRoundingAttack": partial(MyRoundingAttack, eps=0.5, iterations=100),
}


def load_trained_conv_net(with_centering=False):
    conv_net = load_trained_model()
    if with_centering:
        center = CenterImage(cifar10.CHANNEL_MEANS, DEVICE)
        conv_net = nn.Sequential(center, conv_net)
    conv_net.eval()
    return conv_net


def main(attack_name: str = "FAB"):
    model = load_trained_conv_net(with_centering=True)
    train_dl, val_dl = get_data_loader(sizes=(49_000, 1_000), tf="to_tensor")

    print(f"Train accuracy: {get_accuracy(model, train_dl):.1%}")
    print(f"Validation accuracy: {get_accuracy(model, val_dl):.1%}")

    attack = ATTACKS[attack_name](model)

    print(f"Validation loss: {get_loss(model, val_dl, attack):.4f}")

    print(f"\nValidation Results:")
    distances = get_distances(model, val_dl, attack)
    with open(OUTPUTS / f"conv_{attack_name}.txt", "w") as f:
        yaml.safe_dump(distances, f)

    thresholds = [0.1, 0.2, 0.3, 0.5, 1.0]
    for threshold in thresholds:
        adv_acc = sum(d > threshold for d in distances) / len(distances)
        print(f"Adv. acc. for perturbation < {threshold}: {adv_acc:.1%}")


if __name__ == "__main__":
    arguments = sys.argv[1:]
    main(*arguments)


