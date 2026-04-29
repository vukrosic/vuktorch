import numpy as np


class SGD:
    def __init__(self, parameters, lr=0.1):
        self.parameters = list(parameters)
        self.lr = lr

    def zero_grad(self):
        for parameter in self.parameters:
            parameter.grad = np.zeros_like(parameter.grad, dtype=float)

    def step(self):
        for parameter in self.parameters:
            parameter.data -= self.lr * parameter.grad
