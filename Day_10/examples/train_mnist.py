from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vuktorch.datasets import load_mnist
from vuktorch.nn import MLP
from vuktorch.optim import SGD
from vuktorch.train_mnist import evaluate, train_epoch


def main():
    np.random.seed(0)

    (x_train, y_train), (x_test, y_test) = load_mnist(limit_train=2048, limit_test=512)
    model = MLP(784, [128, 64, 10])
    optimizer = SGD(model.parameters(), lr=0.1)

    for epoch in range(3):
        train_loss, train_acc = train_epoch(model, optimizer, x_train, y_train, batch_size=64)
        test_loss, test_acc = evaluate(model, x_test, y_test, batch_size=256)
        print(
            f"epoch={epoch + 1} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.3f} "
            f"test_loss={test_loss:.4f} test_acc={test_acc:.3f}"
        )


if __name__ == "__main__":
    main()
