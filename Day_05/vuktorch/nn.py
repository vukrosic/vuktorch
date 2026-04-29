import numpy as np

from .tensor import Tensor


class Module:
    def parameters(self):
        return []

    def zero_grad(self):
        for parameter in self.parameters():
            parameter.grad = np.zeros_like(parameter.grad, dtype=float)


class Neuron(Module):
    """
    Day 05 keeps the abstraction small: one neuron, one weight vector, one bias.
    """

    def __init__(self, nin, nonlin=True):
        scale = 1.0 / max(1, nin)
        self.w = Tensor(np.random.uniform(-scale, scale, size=(nin,)))
        self.b = Tensor(0.0)
        self.nonlin = nonlin

    def __call__(self, x):
        x = x if isinstance(x, Tensor) else Tensor(x)
        act = (x * self.w).sum() + self.b
        return act.relu() if self.nonlin else act

    def parameters(self):
        return [self.w, self.b]
