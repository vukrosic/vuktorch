# Day 01 - The Scalar

Day 01 is about building the smallest useful abstraction in a deep learning
framework: a scalar value that can participate in a computation graph.

The goal is not speed. The goal is for every line to feel inevitable.

## What You Build

By the end of Day 01, you should have:

- a `Value` class that wraps a single scalar
- support for `+` and `*`
- a graph of parent nodes via `_prev`
- a local backward rule stored on each result node
- a manual understanding of how gradients flow backward

This is the first principles version of autograd before we automate anything.

## Core Idea

A plain Python float can store a number, but it cannot answer:

- where did this number come from?
- what are its parents in the graph?
- what gradient should flow into each parent?

The `Value` class fixes that by storing:

- `data` - the scalar itself
- `grad` - the gradient accumulated at this node
- `_prev` - the nodes that created this one
- `_op` - the operation that produced it
- `_backward` - the local chain-rule step

## Files

- `01_the_scalar_template.ipynb` - teaching notebook with blank code cells for
  recording
- `HOMEWORK.md` - Day 01 challenge for implementing `__pow__`
- `examples/manual_backprop.py` - a tiny worked example showing gradients by
  hand
- `vuktorch/engine.py` - the Day 01 scalar engine

## The Graph We Care About

We use a toy graph like this:

```text
a = 2.0
b = -3.0
c = 10.0
e = a * b
d = e + c
```

Forward values:

- `e = -6.0`
- `d = 4.0`

Backward intuition:

- `dd/dd = 1`
- `dd/de = 1`
- `dd/dc = 1`
- `dd/da = b = -3`
- `dd/db = a = 2`

That is the whole premise of Day 01: each node only needs to know its local
derivative rule.

## Suggested Flow

1. Start with plain Python numbers and show their limits.
2. Create a `Value` wrapper with `data` and `grad`.
3. Implement `__add__`.
4. Implement `__mul__`.
5. Store `_prev` and `_op` so the graph exists.
6. Add a local `_backward` closure on each output node.
7. Manually call those backward steps in the correct order.
8. End by motivating Day 02: automatic traversal with `.backward()`.

## Quick Demo

Run the worked example:

```bash
python3 Day_01/examples/manual_backprop.py
```
