from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vuktorch.nn import MLP
from vuktorch.optim import SGD
from vuktorch.train import make_xor, predict, train_step


def main():
    np.random.seed(0)

    x, y = make_xor()
    model = MLP(2, [8, 8, 1])
    optimizer = SGD(model.parameters(), lr=0.1)

    for step in range(1000):
        loss = train_step(model, optimizer, x, y)
        if step in {0, 99, 199, 499, 999}:
            print(f"step={step:04d} loss={loss:.6f}")

    print("predictions:")
    print(np.round(predict(model, x), 3))


if __name__ == "__main__":
    main()
