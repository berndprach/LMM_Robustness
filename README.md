# Robustness of Large Multimodal Models
Code accompanying the blog post, [Are LMMs Robust to Small Image Perturbations?](https://berndprach.github.io/blog-posts/2025/07/AreLMMsRobust/)


## Usage:
Create local installation of the package:
```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

Run scripts (requires a GPU):
```bash
python lmm_robustness/scripts/evaluate/lmm_accuracy.py
python lmm_robustness/scripts/train_conv_net.py
python lmm_robustness/scripts/evaluate/lmm_adversarial_robustness.py --eps=1.
python lmm_robustness/scripts/evaluate/lmm_adversarial_robustness.py --eps=2.
python lmm_robustness/scripts/evaluate/lmm_adversarial_robustness.py --eps=3.
python lmm_robustness/scripts/plot/lmm_adversarial_robustness.py
```
