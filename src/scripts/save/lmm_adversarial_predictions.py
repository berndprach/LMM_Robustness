import os
import sys
from pathlib import Path

import torch
import yaml

from src.architecture.phi_4_multimodal import (
    load_lmm, get_prompt, generate_responses
)
from src.data import cifar10
from src.data.cifar10 import get_data_loader
from src.scripts.evaluate.conv_robustness import (
    load_trained_conv_net, ATTACKS,
)
from src.scripts.evaluate.lmm_adversarial_robustness import (
    to_pillow, get_distance,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PATH = Path("outputs", "lmm_adversarial_predictions")
os.makedirs(PATH, exist_ok=True)


def get_data_path(idx: int) -> Path:
    return PATH / f"{idx:02d}.yaml"


def main(attack_name: str = "MyAttack", eps_str: str = "1"):
    eps = float(eps_str)

    conv_net = load_trained_conv_net(with_centering=True)
    train_dl, val_dl = get_data_loader(sizes=(49_000, 1_000), tf="to_tensor")
    attack = ATTACKS[attack_name](conv_net, eps=eps*0.99)

    lmm = load_lmm()
    prompts = [get_prompt()]*2

    idx = 0
    for x_batch, y_batch in val_dl:
        x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)
        adv_x_batch = attack(x_batch, y_batch)

        for adx_x, x, y in zip(adv_x_batch, x_batch, y_batch):
            image = to_pillow(x)
            adv_image = to_pillow(adx_x)

            distance = get_distance(image, adv_image)
            predictions = generate_responses([image, adv_image], prompts, lmm)

            data = {
                "path": str(get_data_path(idx)),
                "index": idx,
                "image_path": str(PATH / f"{idx:02d}.png"),
                "adv_image_path": str(PATH / f"{idx:02d}_adv.png"),
                "distance": float(distance),
                "prediction": predictions[0].strip(),
                "adv_prediction": predictions[1].strip(),
                "label": cifar10.CLASS_NAMES[y.item()],
                "attack": attack_name,
                "eps": eps,
            }
            print(data)

            with open(data["path"], "w") as f:
                yaml.safe_dump(data, f)

            image.save(data["image_path"])
            adv_image.save(data["adv_image_path"])

            print(f"Saved data to {data['path']}.")
            idx += 1


if __name__ == "__main__":
    arguments = sys.argv[1:]
    main(*arguments)
