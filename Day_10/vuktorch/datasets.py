from pathlib import Path
from urllib.request import urlretrieve

import numpy as np


MNIST_URL = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"


def load_mnist(path=None, limit_train=None, limit_test=None):
    if path is None:
        path = Path(__file__).resolve().parents[1] / "data" / "mnist.npz"
    else:
        path = Path(path)

    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        urlretrieve(MNIST_URL, path)

    with np.load(path) as data:
        x_train = data["x_train"].reshape(-1, 784).astype(float) / 255.0
        y_train = data["y_train"].astype(int)
        x_test = data["x_test"].reshape(-1, 784).astype(float) / 255.0
        y_test = data["y_test"].astype(int)

    if limit_train is not None:
        x_train = x_train[:limit_train]
        y_train = y_train[:limit_train]

    if limit_test is not None:
        x_test = x_test[:limit_test]
        y_test = y_test[:limit_test]

    return (x_train, y_train), (x_test, y_test)
