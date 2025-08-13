import os
from pathlib import Path

from matplotlib import pyplot as plt

MISTAKE_PATH = Path("outputs", "mistakes")
OUTPUT_PATH = Path("outputs", "mistakes2")
os.makedirs(OUTPUT_PATH, exist_ok=True)


def main():
    for fn in os.listdir(MISTAKE_PATH):
        label, _, prediction, idx_str = fn.replace(".png", "").split("_")
        print(label, prediction, idx_str)

        with open(MISTAKE_PATH / fn, "rb") as f:
            image = plt.imread(f)

        plt.figure(figsize=(6, 3))
        plt.imshow(image)
        plt.title(f"Prediction: {prediction}", y=-0.15, fontsize=18)
        plt.axis("off")
        plt.savefig(OUTPUT_PATH / f"{idx_str}.png", bbox_inches="tight")


if __name__ == "__main__":
    main()
