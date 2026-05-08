from __future__ import annotations

from contextlib import contextmanager
from typing import Callable, Iterable

import numpy as np


_grad_enabled = True


def _ensure_array(data) -> np.ndarray:
    if isinstance(data, Tensor):
        data = data.data
    return np.asarray(data, dtype=np.float64)


def _unbroadcast(grad: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    if shape == ():
        return np.asarray(grad).sum()

    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)

    for axis, size in enumerate(shape):
        if size == 1:
            grad = grad.sum(axis=axis, keepdims=True)
    return grad


@contextmanager
def no_grad():
    global _grad_enabled
    previous = _grad_enabled
    _grad_enabled = False
    try:
        yield
    finally:
        _grad_enabled = previous


class Tensor:
    def __init__(
        self,
        data,
        requires_grad: bool = False,
        _children: Iterable["Tensor"] = (),
        _op: str = "",
    ):
        self.data = _ensure_array(data)
        self.requires_grad = bool(requires_grad)
        self.grad: np.ndarray | None = None
        self._prev = set(_children)
        self._op = _op
        self._backward: Callable[[], None] = lambda: None

    @property
    def shape(self) -> tuple[int, ...]:
        return self.data.shape

    @property
    def ndim(self) -> int:
        return self.data.ndim

    @property
    def T(self) -> "Tensor":
        return self.transpose()

    def item(self) -> float:
        return float(self.data.item())

    def zero_grad(self) -> None:
        self.grad = None

    def detach(self) -> "Tensor":
        return Tensor(self.data.copy())

    def __repr__(self) -> str:
        return f"Tensor(data={self.data}, requires_grad={self.requires_grad})"

    def _requires_grad(self, *others: "Tensor") -> bool:
        return _grad_enabled and (self.requires_grad or any(t.requires_grad for t in others))

    def _add_grad(self, grad: np.ndarray) -> None:
        if self.grad is None:
            self.grad = np.asarray(grad, dtype=np.float64)
        else:
            self.grad = self.grad + grad

    def __add__(self, other) -> "Tensor":
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data, self._requires_grad(other), (self, other), "+")

        def _backward():
            if out.grad is None:
                return
            if self.requires_grad:
                self._add_grad(_unbroadcast(out.grad, self.data.shape))
            if other.requires_grad:
                other._add_grad(_unbroadcast(out.grad, other.data.shape))

        out._backward = _backward
        return out

    def __radd__(self, other) -> "Tensor":
        return self + other

    def __neg__(self) -> "Tensor":
        return self * -1

    def __sub__(self, other) -> "Tensor":
        return self + (-other)

    def __rsub__(self, other) -> "Tensor":
        return other + (-self)

    def __mul__(self, other) -> "Tensor":
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data, self._requires_grad(other), (self, other), "*")

        def _backward():
            if out.grad is None:
                return
            if self.requires_grad:
                self._add_grad(_unbroadcast(other.data * out.grad, self.data.shape))
            if other.requires_grad:
                other._add_grad(_unbroadcast(self.data * out.grad, other.data.shape))

        out._backward = _backward
        return out

    def __rmul__(self, other) -> "Tensor":
        return self * other

    def __truediv__(self, other) -> "Tensor":
        return self * (other if isinstance(other, Tensor) else Tensor(other)) ** -1

    def __rtruediv__(self, other) -> "Tensor":
        return (other if isinstance(other, Tensor) else Tensor(other)) * self**-1

    def __pow__(self, power: float) -> "Tensor":
        out = Tensor(self.data**power, self.requires_grad and _grad_enabled, (self,), f"**{power}")

        def _backward():
            if out.grad is None or not self.requires_grad:
                return
            self._add_grad((power * self.data ** (power - 1)) * out.grad)

        out._backward = _backward
        return out

    def __matmul__(self, other) -> "Tensor":
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data @ other.data, self._requires_grad(other), (self, other), "@")

        def _backward():
            if out.grad is None:
                return
            if self.requires_grad:
                self._add_grad(out.grad @ np.swapaxes(other.data, -1, -2))
            if other.requires_grad:
                other._add_grad(np.swapaxes(self.data, -1, -2) @ out.grad)

        out._backward = _backward
        return out

    def relu(self) -> "Tensor":
        out = Tensor(np.maximum(self.data, 0), self.requires_grad and _grad_enabled, (self,), "relu")

        def _backward():
            if out.grad is None or not self.requires_grad:
                return
            self._add_grad((self.data > 0) * out.grad)

        out._backward = _backward
        return out

    def tanh(self) -> "Tensor":
        t = np.tanh(self.data)
        out = Tensor(t, self.requires_grad and _grad_enabled, (self,), "tanh")

        def _backward():
            if out.grad is None or not self.requires_grad:
                return
            self._add_grad((1 - t**2) * out.grad)

        out._backward = _backward
        return out

    def exp(self) -> "Tensor":
        e = np.exp(self.data)
        out = Tensor(e, self.requires_grad and _grad_enabled, (self,), "exp")

        def _backward():
            if out.grad is None or not self.requires_grad:
                return
            self._add_grad(e * out.grad)

        out._backward = _backward
        return out

    def log(self) -> "Tensor":
        out = Tensor(np.log(self.data), self.requires_grad and _grad_enabled, (self,), "log")

        def _backward():
            if out.grad is None or not self.requires_grad:
                return
            self._add_grad((1 / self.data) * out.grad)

        out._backward = _backward
        return out

    def sum(self, axis=None, keepdims: bool = False) -> "Tensor":
        out = Tensor(self.data.sum(axis=axis, keepdims=keepdims), self.requires_grad and _grad_enabled, (self,), "sum")

        def _backward():
            if out.grad is None or not self.requires_grad:
                return
            grad = out.grad
            if axis is not None and not keepdims:
                axes = (axis,) if isinstance(axis, int) else axis
                for ax in sorted(axes):
                    grad = np.expand_dims(grad, ax)
            self._add_grad(np.ones_like(self.data) * grad)

        out._backward = _backward
        return out

    def mean(self, axis=None, keepdims: bool = False) -> "Tensor":
        if axis is None:
            denom = self.data.size
        elif isinstance(axis, int):
            denom = self.data.shape[axis]
        else:
            denom = np.prod([self.data.shape[i] for i in axis])
        return self.sum(axis=axis, keepdims=keepdims) / denom

    def reshape(self, *shape: int) -> "Tensor":
        if len(shape) == 1 and isinstance(shape[0], tuple):
            shape = shape[0]
        out = Tensor(self.data.reshape(*shape), self.requires_grad and _grad_enabled, (self,), "reshape")

        def _backward():
            if out.grad is None or not self.requires_grad:
                return
            self._add_grad(out.grad.reshape(self.data.shape))

        out._backward = _backward
        return out

    def transpose(self, axes=None) -> "Tensor":
        out = Tensor(np.transpose(self.data, axes), self.requires_grad and _grad_enabled, (self,), "transpose")

        def _backward():
            if out.grad is None or not self.requires_grad:
                return
            if axes is None:
                inverse_axes = None
            else:
                inverse_axes = np.argsort(axes)
            self._add_grad(np.transpose(out.grad, inverse_axes))

        out._backward = _backward
        return out

    def backward(self, grad=None) -> None:
        if grad is None:
            grad = np.ones_like(self.data)
        else:
            grad = _ensure_array(grad)

        topo: list[Tensor] = []
        visited: set[Tensor] = set()

        def build(v: Tensor) -> None:
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build(child)
                topo.append(v)

        build(self)
        self.grad = grad
        for node in reversed(topo):
            node._backward()
