from functools import partial

import torch
import torchattacks
from torch.linalg import vector_norm


def stochastic_rounding(x, step_size):
    x = x / step_size
    noise = torch.rand_like(x) - 0.5
    x = torch.round(x + noise)
    return x * step_size


class MyRoundingAttack:
    def __init__(self, model, norm="L2", eps=10., iterations=10):
        # self.attack = torchattacks.FAB(model, norm=norm, eps=eps)
        self.attack = torchattacks.FAB(model, norm=norm, eps=10.)
        self.loss_function = torch.nn.CrossEntropyLoss()
        self.vector_norm = partial(vector_norm, dim=(1, 2, 3), keepdim=True)
        self.model = model
        self.eps = eps
        self.iterations = iterations

    def __call__(self, x_batch, y_batch):
        x_adv = self.attack(x_batch, y_batch)
        perturbation = x_adv - x_batch
        a = torch.where(
            self.vector_norm(perturbation) > 1e-12,
            self.parameter(perturbation),
            torch.randn_like(perturbation),
        )

        for _ in range(self.iterations):
            a = self.adversarial_step(a, x_batch, y_batch)
            p = self.adversarial_perturbation(a)
            a = self.parameter(stochastic_rounding(p, 1/255))

        return x_batch + self.adversarial_perturbation(a)

    def adversarial_perturbation(self, a: torch.tensor):
        direction = a / self.vector_norm(a)
        magnitude = torch.sin(self.vector_norm(a)) * self.eps
        return direction * magnitude

    def parameter(self, perturbation: torch.Tensor) -> torch.Tensor:
        """ Numerically stable inverse of self.adversarial_perturbation() """
        norm = self.vector_norm(perturbation)
        sine_of_magnitude = norm / self.eps
        sine_of_magnitude = torch.clamp(sine_of_magnitude, -1.0, 1.0)
        a_magnitude = torch.arcsin(sine_of_magnitude)
        a = torch.where(
            norm > 1e-12,
            perturbation / norm * a_magnitude,
            perturbation / self.eps
        )
        return a

    def adversarial_step(self, a, x_batch, y_batch):
        t = torch.zeros_like(a, requires_grad=True)
        x_adv = x_batch + self.adversarial_perturbation(a + t)
        scores = self.model(x_adv)
        loss = self.loss_function(scores, y_batch)

        loss.backward()
        with torch.no_grad():
            a += t.grad * 0.01
        return a
