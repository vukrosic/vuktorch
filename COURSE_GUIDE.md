# VukTorch Course Guide

## Critique

The current repo has a strong Day 01 opening, but the rest of the course is too
thin to function as a real teaching asset.

The main issues are:

- Days 02-10 mostly describe topics, but not the concrete artifact to build in
  each lesson.
- The curriculum does not yet make the transition between days explicit.
  Example: Day 02 introduces `.backward()`, but the bridge from Day 01 manual
  backprop to Day 02 automatic traversal is not spelled out.
- The course lacks a consistent lesson template, so the learning experience
  would feel uneven from day to day.
- Several days combine multiple ideas in one title, but the repo does not state
  which concept is the focus for that specific day.
- There is no visible set of checkpoints or exercises that let the viewer know
  when they have actually understood the material.

## Improvement Plan

The course should be organized around one question per day:

1. What new abstraction do we need?
2. What code artifact proves we built it?
3. What simple exercise checks understanding?

That means every day should have the same internal structure:

- `Goal` - the single lesson objective
- `Build` - the exact artifact to implement
- `Why it matters` - the conceptual bridge from the previous day
- `Exercise` - a small extension task
- `Next day` - how the current lesson sets up the next one

## Course Arc

- Day 01 - Scalars and local derivatives
- Day 02 - Automatic backpropagation over scalar graphs
- Day 03 - Vectorized tensors with NumPy
- Day 04 - Tensor operations, broadcasting, and matrix multiply
- Day 05 - Modules, parameters, and reusable scalar components
- Day 06 - Layers and MLP composition
- Day 07 - Loss functions for regression and classification
- Day 08 - Optimization with SGD
- Day 09 - Training loops and evaluation
- Day 10 - MNIST end-to-end training

## Implementation Rules

- Keep the early days in pure Python until NumPy is genuinely useful.
- Use one conceptual jump per day.
- Keep each lesson artifact small enough to explain live on camera.
- Prefer examples that can be manually checked with pencil-and-paper.
- Make the homework visible and incremental so viewers have a clear extension
  path.

## Current Status

- Day 01 is now a complete teaching packet.
- Day 02 has the autograd engine, but should still get a dedicated demo and
  lesson framing.
- Days 03 and 04 now have source modules and notebook templates.
- Days 05-10 now have notebook templates and lesson briefs.
