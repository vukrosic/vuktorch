import numpy as np


class Tensor:
    """
    Day 03 vectorized tensor with basic autograd.

    This version keeps the math intentionally simple:
    - NumPy arrays as the backing store
    - same-shape elementwise ops
    - reduction ops for losses
    - topological backward traversal
    """

    def __init__(self, data, _children=(), _op="", label=""):
        self.data = np.array(data, dtype=float)
        self.grad = np.zeros_like(self.data, dtype=float)
        self._prev = set(_children)
        self._op = _op
        self.label = label
        self._backward = lambda: None

    def __repr__(self):
        return f"Tensor(data={self.data}, grad={self.grad})"

    @staticmethod
    def _ensure_tensor(other):
        return other if isinstance(other, Tensor) else Tensor(other)

    def __add__(self, other):
        other = self._ensure_tensor(other)
        assert self.data.shape == other.data.shape, "Day 03 keeps elementwise ops same-shaped"
        out = Tensor(self.data + other.data, (self, other), "+")

        def _backward():
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward
        return out

    def __mul__(self, other):
        other = self._ensure_tensor(other)
        assert self.data.shape == other.data.shape, "Day 03 keeps elementwise ops same-shaped"
        out = Tensor(self.data * other.data, (self, other), "*")

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def __pow__(self, other):
        assert isinstance(other, (int, float))
        out = Tensor(self.data**other, (self,), f"**{other}")

        def _backward():
            self.grad += other * (self.data ** (other - 1)) * out.grad

        out._backward = _backward
        return out

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other

    def __rsub__(self, other):
        return other + (-self)

    def sum(self):
        out = Tensor(self.data.sum(), (self,), "sum")

        def _backward():
            self.grad += np.ones_like(self.data) * out.grad

        out._backward = _backward
        return out

    def mean(self):
        return self.sum() * (1.0 / self.data.size)

    def backward(self):
        topo = []
        visited = set()

        def build_topo(node):
            if node in visited:
                return
            visited.add(node)
            for parent in node._prev:
                build_topo(parent)
            topo.append(node)

        build_topo(self)
        self.grad = np.ones_like(self.data, dtype=float)
        for node in reversed(topo):
            node._backward()
