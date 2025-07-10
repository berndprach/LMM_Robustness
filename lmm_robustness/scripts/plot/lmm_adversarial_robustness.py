from matplotlib import pyplot as plt

from lmm_robustness.scripts.evaluate.lmm_adversarial_robustness import (
    RESULTS_PATH
)

plt.rcParams.update({'font.size': 14})


def main():
    with open(RESULTS_PATH, "r") as f:
        lines = f.readlines()

    results = {}
    for line in lines:
        attack_name, eps_str, ra_str = line.strip().split(", ")
        eps = float(eps_str.split("=")[1])
        robust_accuracy = float(ra_str.replace("%", ""))

        if attack_name not in results:
            results[attack_name] = {}
        results[attack_name][eps] = robust_accuracy

    plt.figure(figsize=(6, 3))
    for attack_name, attack_results in results.items():
        perturbations = sorted(attack_results.keys())
        accuracies = [attack_results[eps] for eps in perturbations]

        plt.plot(perturbations, accuracies, label=attack_name, marker='o')

    plt.xlabel("Perturbation Size")
    plt.ylabel("Robust Accuracy")
    plt.title("Adversarial Robustness of LMMs")

    # Remove border lines:
    for direction in ["right", "left"]:
        plt.gca().spines[direction].set_visible(False)

    y_ticks = range(0, 101, 20)
    plt.yticks(y_ticks, [f"{y}%" for y in y_ticks])
    plt.ylim(0, 100)

    # plt.legend()
    plt.grid()

    plt.savefig("outputs/lmm_robustness.png", bbox_inches='tight')
    print("\nSaved plot to outputs/lmm_robustness.png")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
