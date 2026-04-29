import numpy as np

from .losses import cross_entropy
from .tensor import Tensor


def iterate_batches(x, y, batch_size, shuffle=True):
    indices = np.arange(len(x))
    if shuffle:
        np.random.shuffle(indices)

    for start in range(0, len(indices), batch_size):
        batch_indices = indices[start : start + batch_size]
        yield x[batch_indices], y[batch_indices]


def train_epoch(model, optimizer, x, y, batch_size=64):
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for x_batch, y_batch in iterate_batches(x, y, batch_size=batch_size, shuffle=True):
        logits = model(Tensor(x_batch))
        loss = cross_entropy(logits, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += float(loss.data) * len(x_batch)
        total_correct += int((logits.data.argmax(axis=1) == y_batch).sum())
        total_examples += len(x_batch)

    return total_loss / total_examples, total_correct / total_examples


def evaluate(model, x, y, batch_size=256):
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for x_batch, y_batch in iterate_batches(x, y, batch_size=batch_size, shuffle=False):
        logits = model(Tensor(x_batch))
        loss = cross_entropy(logits, y_batch)
        total_loss += float(loss.data) * len(x_batch)
        total_correct += int((logits.data.argmax(axis=1) == y_batch).sum())
        total_examples += len(x_batch)

    return total_loss / total_examples, total_correct / total_examples
