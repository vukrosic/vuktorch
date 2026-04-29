import numpy as np

from .tensor import Tensor


def mse_loss(prediction, target):
    target = target if isinstance(target, Tensor) else Tensor(target)
    return ((prediction - target) ** 2).mean()


def cross_entropy(logits, targets):
    targets = np.array(targets, dtype=int).reshape(-1)
    assert logits.data.ndim == 2, "cross_entropy expects logits with shape (batch, classes)"
    assert logits.data.shape[0] == targets.shape[0], "target count must match batch size"

    shifted = logits.data - logits.data.max(axis=1, keepdims=True)
    exp_scores = np.exp(shifted)
    probs = exp_scores / exp_scores.sum(axis=1, keepdims=True)
    batch_indices = np.arange(targets.shape[0])
    loss_value = -np.log(probs[batch_indices, targets] + 1e-12).mean()
    out = Tensor(loss_value, (logits,), "cross_entropy")

    def _backward():
        grad = probs.copy()
        grad[batch_indices, targets] -= 1.0
        grad /= targets.shape[0]
        logits.grad += grad * out.grad

    out._backward = _backward
    return out
