import numpy as np

from .losses import mse_loss
from .tensor import Tensor


def make_xor():
    x = np.array(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
        ]
    )
    y = np.array([[-1.0], [1.0], [1.0], [-1.0]])
    return x, y


def train_step(model, optimizer, x, y):
    predictions = model(Tensor(x))
    loss = mse_loss(predictions, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return float(loss.data)


def predict(model, x):
    return model(Tensor(x)).data
