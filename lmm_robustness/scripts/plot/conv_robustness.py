import os

import yaml
from matplotlib import pyplot as plt

from lmm_robustness.scripts.evaluate.conv_robustness import OUTPUTS


def save_robustness_plot():
    for fn in os.listdir(OUTPUTS):
        with open(OUTPUTS / fn, "r") as f:
            distances = yaml.safe_load(f)

        attack_name = fn.split("_")[1].split(".")[0]
        draw_robustness_plot(distances, attack_name)

    plt.legend()
    plt.grid()

    plt.savefig(OUTPUTS / "robustness_plot.png", bbox_inches='tight')
    plt.show()


def draw_robustness_plot(distances: list[float], label=None):
    distances.sort()
    remaining_acc = [1 - (i+1) / len(distances) for i in range(len(distances))]

    plt.plot(distances, remaining_acc, label=label)

    plt.title("Empirical Robustness Plot")
    plt.xlabel("Perturbation size")
    plt.ylabel("Remaining Accuracy")

    plt.ylim(0., 1.)
    plt.xlim(-0., 0.5)

    y_ticks = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    plt.yticks(y_ticks, [f"{y:.0%}" for y in y_ticks])
