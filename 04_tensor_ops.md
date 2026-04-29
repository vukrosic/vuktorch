# Day 04: Tensor Ops

**Series:** PyTorch from Scratch: The Zero-to-One Framework

## Goal

Add the tensor operations real models need: broadcasting, matrix multiplication, and a few utility ops.

## What You Should Understand By The End

- why same-shape elementwise math is not enough
- what broadcasting changes in the backward pass
- how matrix multiplication drives dense layers
- why shape logic matters as much as numeric logic

## Why Day 04 Exists

Day 03 could handle vectors and reductions, but it still assumed shapes matched exactly. Real neural network code depends on:

- adding biases across batches
- multiplying batches by weight matrices
- applying nonlinearities and reductions after that

That means shape-aware backward rules now matter.

## Step 1: Add Broadcasting-Aware Addition

The forward pass for broadcasting is easy because NumPy already does it.

```python
out = Tensor(self.data + other.data, (self, other), "+")
```

The real issue is backward. A broadcasted tensor may need its gradient summed across one or more axes before it matches the original shape again.

One useful helper is: "sum gradients over expanded dimensions until the shape matches the source tensor."

Question: if a bias of shape `(3,)` is added to a batch of shape `(5, 3)`, what shape should the bias gradient have?

## Step 2: Add Broadcasting-Aware Multiplication

Multiplication has the same shape problem as addition.

Forward:

```python
out = Tensor(self.data * other.data, (self, other), "*")
```

Backward before shape fixing:

```python
self_grad = other.data * out.grad
other_grad = self.data * out.grad
```

If either side was broadcasted, reduce the gradient back to that side's original shape before accumulating.

The math is not the hard part here. The shape bookkeeping is.

## Step 3: Add Matrix Multiplication

Matrix multiplication is the heart of dense layers.

```python
out = Tensor(self.data @ other.data, (self, other), "@")
```

For a simple 2D case:

```python
def _backward():
    self.grad += out.grad @ other.data.T
    other.grad += self.data.T @ out.grad
```

This is one of the most important backward rules in the whole course. Once this works, layers become straightforward.

Question: why does the backward rule for the weights use `self.data.T @ out.grad`?

## Step 4: Add ReLU

```python
out = Tensor(np.maximum(0, self.data), (self,), "relu")
```

Backward:

```python
self.grad += (self.data > 0) * out.grad
```

ReLU is conceptually simple, which is useful because matrix multiplication already added a lot of shape complexity.

## Step 5: Test A Tiny Batch Example

```python
x = Tensor([[1.0, 2.0],
            [3.0, 4.0]])
w = Tensor([[2.0, 0.0],
            [0.0, 2.0]])

y = (x @ w).sum()
y.backward()
```

Questions:

- what shape should `x @ w` have?
- what shape should `x.grad` have?
- what shape should `w.grad` have?

If the shapes are wrong, the gradients are wrong, even if the code runs.

## Homework

### Homework 1: Add Bias Broadcasting

Test a case like this:

```python
x = Tensor([[1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0]])
b = Tensor([0.1, 0.2, 0.3])
y = (x + b).sum()
y.backward()
```

Expected idea:

- `x.grad` should be all ones
- `b.grad` should be `[2.0, 2.0, 2.0]`

This is good homework because it forces broadcasting to work in backward, not just forward.

### Homework 2: Add `matmul` Tests

Write one test that checks output shape and one that checks gradients against a hand-computed tiny example.

### Stretch: Batched Matmul

If the 2D case is stable, think about what would need to change for higher-rank batches.

## End Of Day 04

Day 04 is complete once tensor shapes stop feeling incidental and start feeling like part of the autograd design.
