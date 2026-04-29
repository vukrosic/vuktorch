# Day 04 - Tensor Ops

Day 04 upgrades the tensor engine with the operations neural networks actually
need.

## Goal

Add broadcasting rules, matrix multiplication, and a few more tensor utilities.

## Build

- broadcasting-aware elementwise ops
- matrix multiplication
- ReLU and reduction helpers

## Why It Matters

This is the point where the framework becomes expressive enough for linear
layers and batched data.

## Exercise

Multiply a batch matrix by a weight matrix and confirm the output shape and
gradient flow.

## Files

- `04_tensor_ops_template.ipynb`
- `vuktorch/tensor.py`
