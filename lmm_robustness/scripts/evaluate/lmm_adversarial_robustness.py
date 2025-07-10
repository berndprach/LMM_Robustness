import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torchvision.transforms import transforms as tfs

from lmm_robustness.architecture.center import CenterImage
from lmm_robustness.data import cifar10
from lmm_robustness.data.cifar10 import get_data_loader
from lmm_robustness.scripts.evaluate.conv_robustness import (
    load_trained_model, ATTACKS
)
from lmm_robustness.scripts.evaluate.lmm_accuracy import (
    classify
)
from lmm_robustness.architecture.phi_4_multimodal import load_lmm, get_prompt
from lmm_robustness.scripts.train_conv_net import STATE_DICT_PATH

# STATE_DICT_PATH = "outputs/state_dict.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RESULTS_PATH = Path("outputs", "results.txt")
os.makedirs(RESULTS_PATH.parent, exist_ok=True)


def to_pillow(x: torch.Tensor) -> Image.Image:
    """Convert a tensor to a PIL Image."""
    x = torch.clamp(x, 0, 1)
    x = x.permute(1, 2, 0).detach().cpu().numpy()
    x = (x * 255).astype('uint8')
    return Image.fromarray(x)


def get_distance(p: Image.Image, q: Image.Image) -> float:
    """Calculate the distance between two PIL Images."""
    p_arr = np.array(p, dtype=np.int64).flatten()
    q_arr = np.array(q, dtype=np.int64).flatten()

    return np.linalg.norm(p_arr - q_arr, ord=2) / 255


def main(attack_name: str = "MyAttack", eps_str: str = "1"):
    eps = float(eps_str)

    conv_net = load_trained_model()
    center = CenterImage(cifar10.CHANNEL_MEANS, DEVICE)
    conv_net = nn.Sequential(center, conv_net)
    conv_net.eval()

    train_dl, val_dl = get_data_loader(sizes=(49_000, 1_000), tf="to_tensor")
    attack = ATTACKS[attack_name](conv_net, eps=eps*0.99)

    # attack = torchattacks.FAB(conv_net, norm="L2", eps=10.)

    # attack = MyAttack(conv_net, norm="L2", eps=1.)
    # eps=1 => Correct: 44/50 (88.0%), Robust: 34/50 (68.0%)
    # eps=2 => Correct: 44/50 (88.0%), Robust: 15/50 (30.0%)
    # eps=10 => Correct: 46/50 (92.0%), Robust: 7/50 (14.0%)

    # attack = MyRoundingAttack(conv_net, norm="L2", eps=10., iterations=100)
    # eps=1 => Correct: 42/50 (84.0%), Robust: 34/50 (68.0%)
    # eps=2 => Correct: 48/50 (96.0%), Robust: 12/50 (24.0%)
    # eps=10 => Correct: 105/120 (87.5%), Robust: 7/120 (5.8%)

    model = load_lmm()
    prompt = get_prompt()

    correct, robust, total = 0, 0, 0
    for x_batch, y_batch in val_dl:
        x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)
        adv_x_batch = attack(x_batch, y_batch)

        for adx_x, x, y in zip(adv_x_batch, x_batch, y_batch):
            image = to_pillow(x)
            adv_image = to_pillow(adx_x)

            if get_distance(image, adv_image) > eps:
                print(f"Skipping large perturbation "
                      f"({get_distance(image, adv_image):.2g}).")
                total += 1
                robust += 1
                continue

            predictions = classify([adv_image], model, [prompt])
            robust += int(predictions[0] == y.item())
            total += 1

            print(f"Robust: {robust}/{total} ({robust/total:.1%})", end='\r')

    with open(RESULTS_PATH, "a") as f:
        f.write(f"{attack_name}, eps={eps_str}, {robust/total:.1%}\n")

    # plot_results()


if __name__ == "__main__":
    arguments = sys.argv[1:]
    main(*arguments)
