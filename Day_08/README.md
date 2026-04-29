# Day 08 - Optimization

Day 08 closes the loop between gradients and parameter updates.

## Goal

Add an `SGD` optimizer and show how parameter updates close the training loop.

## Build

- `SGD`
- `zero_grad`
- parameter update step

## Why It Matters

Backpropagation gives direction. Optimization turns that direction into learning
by nudging parameters after each backward pass.

## Exercise

Run one gradient descent step on a tiny example and confirm the loss decreases.
