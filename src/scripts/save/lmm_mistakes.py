import os
from pathlib import Path
from typing import Iterable

import torch
from matplotlib import pyplot as plt

from src.data import cifar10
from src.data.batch import batch
from src.architecture.phi_4_multimodal import (
    load_lmm, get_prompt, generate_responses
)

BATCH_SIZE = 4
MISTAKE_PATH = Path("outputs", "mistakes")
os.makedirs(MISTAKE_PATH, exist_ok=True)


def add_responses(data: Iterable, prompts, model):
    batch_size = len(prompts)
    for images, labels in batch(data, batch_size):
        with torch.no_grad():
            responses = generate_responses(images, prompts, model)
        for image, label, response in zip(images, labels, responses):
            yield image, label, response.strip()


def main():
    lmm = load_lmm()
    prompts = [get_prompt()] * BATCH_SIZE

    _, validation_data = cifar10.get_data(sizes=(49_000, 1_000))

    idx = 0
    for image, label, response in add_responses(validation_data, prompts, lmm):
        if label != cifar10.CLASS_ID.get(response, -1):
            print(f'❌ Misclassification: {response} (label {label}) [{idx}]')
            fn = f"{cifar10.CLASS_NAMES[label]}_called_{response}_{idx}.png"

            # image.save(MISTAKE_PATH / fn)

            plt.figure(figsize=(6, 3))
            plt.imshow(image)
            plt.title(f"Prediction: {response}", y=-0.15, fontsize=18)
            plt.axis("off")
            plt.savefig(MISTAKE_PATH / fn, bbox_inches="tight")

            print(f"Saved to {MISTAKE_PATH / fn}")
            idx += 1


if __name__ == "__main__":
    main()
