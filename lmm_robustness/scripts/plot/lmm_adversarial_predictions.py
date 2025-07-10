import os

import yaml
from PIL import Image
from matplotlib import pyplot as plt

from lmm_robustness.scripts.save import lmm_adversarial_predictions

COMBINATIONS_PATH = lmm_adversarial_predictions.PATH / "combined"
os.makedirs(COMBINATIONS_PATH, exist_ok=True)

plt.rcParams.update({'font.size': 14})


def main():
    idx = 0
    while True:
        combine(idx)
        idx += 1


def combine(idx: int):
    data_path = lmm_adversarial_predictions.get_data_path(idx)
    with open(data_path, "r") as f:
        data = yaml.safe_load(f)
    # print(f"Data for index {idx}:\n{data}")

    image = Image.open(data["image_path"])
    adv_image = Image.open(data["adv_image_path"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 3))
    fig.canvas.manager.set_window_title(str(data_path))

    ax1.imshow(image)
    ax1.set_title("Prediction: " + data["prediction"], y=-0.15, fontsize=18)
    ax1.axis("off")

    ax2.imshow(adv_image)
    ax2.set_title("Prediction: " + data["adv_prediction"], y=-0.15, fontsize=18)
    ax2.axis("off")

    fn = str(data["index"]) + ".png"
    plt.savefig(COMBINATIONS_PATH / fn, bbox_inches="tight")
    plt.close()

    if data["label"] == data["prediction"] != data["adv_prediction"]:
        print(html_code(data["index"]))


def html_code(idx):
    return (f"<img src=\"/images/blog-posts/LMMRobustness/adv-ex/{idx}.png\" "
            f"alt=\"Adversarial Example\" width=\"300\" hspace=\"20\"/>")


def latex_code(idx):
    return (r"\includegraphics[width = 0.45\textwidth]{Images/combined/"
            + str(idx) + r"} \hfill")


if __name__ == "__main__":
    main()
