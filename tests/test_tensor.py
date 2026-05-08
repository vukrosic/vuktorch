import numpy as np

from vuktorch import Tensor


def test_scalar_backward():
    x = Tensor(2.0, requires_grad=True)
    y = Tensor(-3.0, requires_grad=True)

    z = x * y + x**2
    z.backward()

    assert np.allclose(z.data, -2.0)
    assert np.allclose(x.grad, 1.0)
    assert np.allclose(y.grad, 2.0)


def test_broadcast_backward():
    x = Tensor(np.ones((2, 3)), requires_grad=True)
    b = Tensor(np.array([1.0, 2.0, 3.0]), requires_grad=True)

    y = (x + b).sum()
    y.backward()

    assert np.allclose(x.grad, np.ones((2, 3)))
    assert np.allclose(b.grad, np.array([2.0, 2.0, 2.0]))


def test_matmul_backward_shapes():
    x = Tensor(np.ones((4, 3)), requires_grad=True)
    w = Tensor(np.ones((3, 2)), requires_grad=True)

    y = (x @ w).mean()
    y.backward()

    assert x.grad.shape == (4, 3)
    assert w.grad.shape == (3, 2)
