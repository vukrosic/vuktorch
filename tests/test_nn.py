import numpy as np

from vuktorch import MLP, SGD, Tensor, cross_entropy, mse_loss


def test_linear_model_can_reduce_mse():
    np.random.seed(0)
    x = np.random.randn(32, 1)
    y = 3 * x + 1

    model = MLP(1, [], 1)
    optim = SGD(model.parameters(), lr=0.1)

    first_loss = None
    last_loss = None
    for _ in range(80):
        pred = model(Tensor(x))
        loss = mse_loss(pred, y)
        if first_loss is None:
            first_loss = loss.item()
        optim.zero_grad()
        loss.backward()
        optim.step()
        last_loss = loss.item()

    assert last_loss < first_loss * 0.1


def test_cross_entropy_has_correct_gradient_shape():
    logits = Tensor([[2.0, 0.1, -1.0], [0.2, 1.0, -0.5]], requires_grad=True)
    loss = cross_entropy(logits, np.array([0, 1]))

    loss.backward()

    assert loss.item() > 0
    assert logits.grad.shape == logits.data.shape
    assert np.allclose(logits.grad.sum(axis=1), np.zeros(2))
