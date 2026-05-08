from __future__ import annotations

import numpy as np

from .tensor import Tensor


def mse_loss(pred: Tensor, target) -> Tensor:
    target = target if isinstance(target, Tensor) else Tensor(target)
    return ((pred - target) ** 2).mean()


def cross_entropy(logits: Tensor, targets) -> Tensor:
    targets = np.asarray(targets, dtype=np.int64)
    if logits.data.ndim != 2:
        raise ValueError("cross_entropy expects logits with shape (batch, classes)")
    if targets.shape != (logits.data.shape[0],):
        raise ValueError("targets must have shape (batch,)")

    shifted = logits.data - logits.data.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    probs = exp / exp.sum(axis=1, keepdims=True)
    n = logits.data.shape[0]
    loss_value = -np.log(probs[np.arange(n), targets]).mean()
    out = Tensor(loss_value, requires_grad=logits.requires_grad, _children=(logits,), _op="cross_entropy")

    def _backward():
        if out.grad is None or not logits.requires_grad:
            return
        grad = probs.copy()
        grad[np.arange(n), targets] -= 1
        grad = grad / n
        logits._add_grad(grad * out.grad)

    out._backward = _backward
    return out
