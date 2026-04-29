# Day 02 - Autograd

Day 02 is the bridge from manual chain rule to an actual autograd engine.

## Goal

Automate backpropagation by traversing the scalar computation graph in
topological order and calling each node's local backward rule.

## Build

- `.backward()` on `Value`
- topological sorting over the scalar graph
- gradient accumulation with `+=`

## Why It Matters

Day 01 teaches the idea of local derivatives. Day 02 turns that idea into a
reusable mechanism so every later day can rely on gradients instead of manual
derivative bookkeeping.

## Exercise

Add a small demo script that builds `d = a * b + c`, runs `.backward()`, and
prints the gradients.

## Files

- `02_autograd_template.ipynb`
- `examples/autograd_demo.py`
- `vuktorch/engine.py`
