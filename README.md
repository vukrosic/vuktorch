# Tiny PyTorch From Scratch: AI Research Portfolio Project

`VukTorch` is a tiny PyTorch-style deep learning framework built from scratch.

The goal is not to recreate all of PyTorch. The goal is to build the core ideas
that matter for AI/ML interviews and research engineering: tensors, reverse-mode
autograd, modules, losses, optimizers, and training loops.

Community: [Become AI Researcher](https://skool.com/become-ai-researcher-2669/about)

## Portfolio Outcome

By the end, this repo is meant to read like a job-application portfolio project:

- a working NumPy-backed deep learning framework
- tests that prove core gradients, broadcasting, matrix multiplication, losses, and training behavior
- examples that train small models end to end
- interview questions, resume bullets, and extension ideas in `portfolio/`
- a clear story you can explain in an ML engineering interview: "I rebuilt the core training stack from first principles"

Strong portfolio signal comes from being able to point at both the code and the
reasoning. The lessons explain the design choices, while `vuktorch/`, `tests/`,
and `examples/` prove that the ideas run.

## Curriculum

- [Day 01: The Scalar](01_the_scalar.md)
- [Day 02: Autograd](02_autograd.md)
- [Day 03: Tensors](03_tensors.md)
- [Day 04: Tensor Ops](04_tensor_ops.md)
- [Day 05: Modules and Linear Layers](05_modules_neurons.md)
- [Day 06: Sequential Models and MLPs](06_layers_mlps.md)
- [Day 07: Loss Functions](07_losses.md)
- [Day 08: Optimization](08_optimization.md)
- [Day 09: Training Loops](09_training_loops.md)
- [Day 10: End-to-End Classifier](10_mnist.md)

## Structure

- Each day is a single markdown lesson file at the repo root.
- `vuktorch/` contains the framework code.
- `tests/` contains correctness tests.
- `examples/` contains runnable demos.
- `portfolio/` contains interview, resume, README, and extension assets.
- `plans/` contains the YouTube and Skool production plan.

The course is intentionally incremental: each day adds only the code needed for
that lesson and leaves out repo meta-planning that does not belong in the
teaching material.

## Quickstart

```bash
python -m pip install -e ".[dev]"
pytest
python examples/scalar_demo.py
python examples/mlp_moons.py
```

## What Is Implemented

- NumPy-backed `Tensor`
- reverse-mode autodiff
- broadcasting-aware gradients
- matrix multiplication gradients
- reductions and activations
- `Module`, `Parameter`, `Linear`, `Sequential`, `MLP`
- MSE and cross entropy losses
- SGD and Adam optimizers

## Portfolio Checklist

- Run the tests with `pytest`.
- Run the scalar demo with `python examples/scalar_demo.py`.
- Train the end-to-end classifier with `python examples/mlp_moons.py`.
- Customize the materials in `portfolio/` for your resume, interviews, and README.

Continue with the community here: [Become AI Researcher](https://skool.com/become-ai-researcher-2669/about).
