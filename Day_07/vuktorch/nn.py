import numpy as np

from .tensor import Tensor


class Module:
    def parameters(self):
        return []

    def zero_grad(self):
        for parameter in self.parameters():
            parameter.grad = np.zeros_like(parameter.grad, dtype=float)


class Linear(Module):
    def __init__(self, in_features, out_features):
        scale = 1.0 / np.sqrt(max(1, in_features))
        self.weight = Tensor(np.random.uniform(-scale, scale, size=(in_features, out_features)))
        self.bias = Tensor(np.zeros((1, out_features)))

    def __call__(self, x):
        x = x if isinstance(x, Tensor) else Tensor(x)
        return (x @ self.weight) + self.bias

    def parameters(self):
        return [self.weight, self.bias]


class MLP(Module):
    def __init__(self, in_features, widths):
        sizes = [in_features] + list(widths)
        self.layers = [Linear(sizes[i], sizes[i + 1]) for i in range(len(widths))]

    def __call__(self, x):
        x = x if isinstance(x, Tensor) else Tensor(x)
        for index, layer in enumerate(self.layers):
            x = layer(x)
            if index != len(self.layers) - 1:
                x = x.relu()
        return x

    def parameters(self):
        params = []
        for layer in self.layers:
            params.extend(layer.parameters())
        return params
