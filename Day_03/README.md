# Day 03 - Tensors

Day 03 is the first vectorized version of the framework.

## Goal

Move from single scalars to multidimensional arrays and introduce NumPy only
when it becomes useful.

## Build

- a `Tensor` class backed by NumPy arrays
- same-shape elementwise math
- reduction operations like `sum` and `mean`

## Why It Matters

Scalars are enough to learn autograd, but tensors are what let the framework
handle real data efficiently.

## Exercise

Create two same-shaped tensors, add them, multiply them, and verify the
gradients by hand on a tiny example.

## Files

- `03_tensors_template.ipynb`
- `vuktorch/tensor.py`
