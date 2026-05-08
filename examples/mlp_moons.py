import numpy as np

from vuktorch import MLP, SGD, Tensor, cross_entropy


def make_moons(n: int = 200, noise: float = 0.08, seed: int = 42):
    rng = np.random.default_rng(seed)
    n1 = n // 2
    n2 = n - n1
    t1 = np.linspace(0, np.pi, n1)
    t2 = np.linspace(0, np.pi, n2)
    x1 = np.c_[np.cos(t1), np.sin(t1)]
    x2 = np.c_[1 - np.cos(t2), 1 - np.sin(t2) - 0.5]
    x = np.vstack([x1, x2]) + rng.normal(scale=noise, size=(n, 2))
    y = np.array([0] * n1 + [1] * n2)
    return x, y


def main():
    np.random.seed(0)
    x, y = make_moons()
    model = MLP(2, [8, 8], 2)
    optim = SGD(model.parameters(), lr=0.02)

    for step in range(301):
        logits = model(Tensor(x))
        loss = cross_entropy(logits, y)

        optim.zero_grad()
        loss.backward()
        optim.step()
        for parameter in model.parameters():
            parameter.data = np.clip(parameter.data, -5.0, 5.0)

        if step % 100 == 0:
            preds = logits.data.argmax(axis=1)
            acc = (preds == y).mean()
            print(f"step={step:03d} loss={loss.item():.4f} acc={acc:.3f}")


if __name__ == "__main__":
    main()
