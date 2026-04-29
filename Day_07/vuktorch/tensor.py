import numpy as np


def _unbroadcast(grad, shape):
    if shape == ():
        return np.array(grad).sum()

    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)

    for axis, size in enumerate(shape):
        if size == 1:
            grad = grad.sum(axis=axis, keepdims=True)

    return grad.reshape(shape)


class Tensor:
    """
    Day 04 tensor with broadcasting and matrix multiply.
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
        out = Tensor(self.data + other.data, (self, other), "+")

        def _backward():
            self.grad += _unbroadcast(out.grad, self.data.shape)
            other.grad += _unbroadcast(out.grad, other.data.shape)

        out._backward = _backward
        return out

    def __mul__(self, other):
        other = self._ensure_tensor(other)
        out = Tensor(self.data * other.data, (self, other), "*")

        def _backward():
            self.grad += _unbroadcast(other.data * out.grad, self.data.shape)
            other.grad += _unbroadcast(self.data * out.grad, other.data.shape)

        out._backward = _backward
        return out

    def __pow__(self, other):
        assert isinstance(other, (int, float))
        out = Tensor(self.data**other, (self,), f"**{other}")

        def _backward():
            self.grad += other * (self.data ** (other - 1)) * out.grad

        out._backward = _backward
        return out

    def __matmul__(self, other):
        other = self._ensure_tensor(other)
        assert self.data.ndim == 2 and other.data.ndim == 2, "Day 04 matmul is 2D"
        out = Tensor(self.data @ other.data, (self, other), "@")

        def _backward():
            self.grad += out.grad @ other.data.T
            other.grad += self.data.T @ out.grad

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

    def __truediv__(self, other):
        other = self._ensure_tensor(other)
        return self * (other**-1)

    def __rtruediv__(self, other):
        other = self._ensure_tensor(other)
        return other * (self**-1)

    def sum(self, axis=None, keepdims=False):
        out = Tensor(self.data.sum(axis=axis, keepdims=keepdims), (self,), "sum")

        def _backward():
            grad = out.grad
            if axis is None:
                self.grad += np.ones_like(self.data) * grad
            else:
                if not keepdims:
                    if isinstance(axis, tuple):
                        for ax in sorted(axis):
                            grad = np.expand_dims(grad, axis=ax)
                    else:
                        grad = np.expand_dims(grad, axis=axis)
                self.grad += np.ones_like(self.data) * grad

        out._backward = _backward
        return out

    def mean(self, axis=None, keepdims=False):
        if axis is None:
            denom = self.data.size
        else:
            axes = (axis,) if isinstance(axis, int) else axis
            denom = np.prod([self.data.shape[ax] for ax in axes])
        return self.sum(axis=axis, keepdims=keepdims) * (1.0 / denom)

    def relu(self):
        out = Tensor(np.maximum(self.data, 0.0), (self,), "relu")

        def _backward():
            self.grad += (self.data > 0).astype(float) * out.grad

        out._backward = _backward
        return out

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
