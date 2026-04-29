# Day 03 - Tensors

Day 03 introduces the first real tensor abstraction.

## Goal

Move from single scalars to multidimensional arrays and introduce NumPy only
when vectorization starts paying off.

## Build

- a tensor `Value`-like object that stores NumPy arrays
- shape-aware data handling
- basic elementwise operations

## Why It Matters

Scalars are enough to teach autograd, but tensors are what make the framework
practical. This day is the first step toward batching and vectorized math.

## Exercise

Create a tensor that can add and multiply elementwise, then verify the result
against a handwritten NumPy example.
