import argparse
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torchvision.transforms import transforms as tfs
from torchvision.transforms import ToPILImage  # built-in function

from lmm_robustness.architecture.center import CenterImage
from lmm_robustness.data import cifar10
from lmm_robustness.data.cifar10 import get_data_loader
from lmm_robustness.scripts.evaluate.conv_robustness import (
    load_trained_model, ATTACKS
)
from lmm_robustness.scripts.evaluate.lmm_accuracy import classify
from lmm_robustness.architecture.phi_4_multimodal import load_lmm, get_prompt

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RESULTS_PATH = Path("outputs", "results.txt")
os.makedirs(RESULTS_PATH.parent, exist_ok=True)


parser = argparse.ArgumentParser()
parser.add_argument("--attack_name", type=str, default="GARounding")
parser.add_argument("--eps", type=float, default=1.)
parser.add_argument("-n", "--val-size", type=int, default=1000)


def main():
    args = parser.parse_args()

    conv_net = load_trained_model()
    center = CenterImage(cifar10.CHANNEL_MEANS, DEVICE)
    conv_net = nn.Sequential(center, conv_net)
    conv_net.eval()

    _, val_dl = get_data_loader(sizes=(1, args.val_size), tf="to_tensor")
    attack = ATTACKS[args.attack_name](conv_net, eps=args.eps*0.99)

    model = load_lmm()
    prompt = get_prompt()

    correct, robust, total = 0, 0, 0
    for x_batch, y_batch in val_dl:
        x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)
        adv_x_batch = attack(x_batch, y_batch)

        for adx_x, x, y in zip(adv_x_batch, x_batch, y_batch):
            image = to_pillow(x)
            adv_image = to_pillow(adx_x)

            d = get_distance(image, adv_image)
            if d > args.eps:
                print(f"Skipping large perturbation (distance {d:.3g}).")
                total += 1
                robust += 1
                continue

            predictions = classify([adv_image], model, [prompt])
            robust += int(predictions[0] == y.item())
            total += 1

            print(f"Robust: {robust}/{total} ({robust/total:.1%})", end='\r')

    print(f"\nRobustness for eps={args.eps}: {robust/total:.1%}.")

    with open(RESULTS_PATH, "a") as f:
        f.write(f"{args.attack_name}, eps={args.eps}, {robust/total:.1%}\n")


def to_pillow(x: torch.Tensor) -> Image.Image:
    """Convert a tensor to a PIL Image."""
    # x = torch.clamp(x, 0, 1)
    # x = x.permute(1, 2, 0).detach().cpu().numpy()
    # x = (x * 255).astype('uint8')
    # return Image.fromarray(x)
    return ToPILImage()(torch.clamp(x, 0, 1))


def get_distance(p: Image.Image, q: Image.Image) -> float:
    """Calculate the distance between two PIL Images."""
    p_arr = np.array(p, dtype=np.int64).flatten()
    q_arr = np.array(q, dtype=np.int64).flatten()

    return np.linalg.norm(p_arr - q_arr, ord=2) / 255


if __name__ == "__main__":
    main()
