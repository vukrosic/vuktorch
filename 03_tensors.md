# Day 03: Tensors

**Series:** PyTorch from Scratch: The Zero-to-One Framework

## Goal

Move from scalar values to NumPy-backed tensors while keeping the same autograd ideas.

## What You Should Understand By The End

- why scalars are too limited for real data
- how a tensor is still just data plus gradient history
- how elementwise operations work on same-shaped arrays
- how reductions like `sum` and `mean` turn many numbers into one

## Why Day 03 Exists

Scalars are enough to learn autograd, but real models do not consume one number at a time. They work on vectors, matrices, and batches.

The conceptual jump is smaller than it looks. A tensor is still:

- data
- gradient
- parent references
- a local backward rule

The only real difference is that `data` is now an array.

## Step 1: Wrap A NumPy Array

```python
import numpy as np

class Tensor:
    def __init__(self, data, _children=(), _op="", label=""):
        self.data = np.array(data, dtype=float)
        self.grad = np.zeros_like(self.data)
        self._prev = set(_children)
        self._op = _op
        self.label = label
        self._backward = lambda: None
```

`grad` now matches the shape of `data`. That is the core tensor generalization of the scalar engine.

Question: why can `grad` no longer be a single float?

## Step 2: Add Elementwise Addition

```python
def __add__(self, other):
    other = other if isinstance(other, Tensor) else Tensor(other)
    out = Tensor(self.data + other.data, (self, other), "+")
```

The forward pass is just NumPy addition.

```python
    def _backward():
        self.grad += out.grad
        other.grad += out.grad
```

For same-shaped tensors, the addition backward rule is identical to the scalar case, just applied elementwise.

```python
    out._backward = _backward
    return out
```

## Step 3: Add Elementwise Multiplication

```python
def __mul__(self, other):
    other = other if isinstance(other, Tensor) else Tensor(other)
    out = Tensor(self.data * other.data, (self, other), "*")
```

Again, the forward pass is just NumPy.

```python
    def _backward():
        self.grad += other.data * out.grad
        other.grad += self.data * out.grad
```

This is the same local rule as the scalar product, but now it happens across every position in the array.

```python
    out._backward = _backward
    return out
```

Question: how is this rule similar to Day 01, and what changed?

## Step 4: Add A Power Operation

Power is useful because it lets you write squares and other simple nonlinear expressions directly.

```python
def __pow__(self, power):
    assert isinstance(power, (int, float))
    out = Tensor(self.data ** power, (self,), f"**{power}")

    def _backward():
        self.grad += power * (self.data ** (power - 1)) * out.grad

    out._backward = _backward
    return out
```

This is still the same pattern: forward computation first, local derivative rule second.

## Step 5: Add Reductions

A reduction turns many numbers into fewer numbers. The simplest one is `sum`.

```python
def sum(self):
    out = Tensor(self.data.sum(), (self,), "sum")

    def _backward():
        self.grad += np.ones_like(self.data) * out.grad

    out._backward = _backward
    return out
```

Every input position contributes equally to a sum, so every position receives the output gradient.

Now add `mean` on top of `sum`.

```python
def mean(self):
    return self.sum() * (1.0 / self.data.size)
```

This is a good design because it reuses existing operations instead of introducing new derivative logic.

## Step 6: Test A Tiny Example

```python
x = Tensor([1.0, 2.0, 3.0])
y = Tensor([4.0, 5.0, 6.0])
z = (x * y).sum()
z.backward()
```

Expected gradients:

- `x.grad = [4.0, 5.0, 6.0]`
- `y.grad = [1.0, 2.0, 3.0]`

Question: why does the `sum()` make `z` a scalar-like output again?

## Homework

### Homework 1: Add `mean()`

If `sum()` works, implement `mean()` and verify its gradient on a short vector.

```python
x = Tensor([2.0, 4.0])
y = x.mean()
y.backward()
```

Expected gradient:

- `x.grad = [0.5, 0.5]`

### Homework 2: Add `relu()`

Implement elementwise ReLU:

```python
np.maximum(0, self.data)
```

Backward rule:

- pass gradient where `self.data > 0`
- block gradient where `self.data <= 0`

Test:

```python
x = Tensor([-1.0, 2.0, 0.0])
y = x.relu().sum()
y.backward()
```

Expected gradient:

- `[0.0, 1.0, 0.0]`

### Stretch: Add `__truediv__`

Reuse `__pow__` to support division.

## End Of Day 03

Day 03 is complete once arrays feel like a direct extension of the scalar engine instead of a brand new system.
