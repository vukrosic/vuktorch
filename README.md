# VukTorch

`PyTorch from Scratch: The Zero-to-One Framework`

This repository is structured as a 10-part masterclass for building a small
deep learning framework from first principles in a Karpathy-style,
readability-first approach.

The full teaching strategy and critique of the curriculum live in
[COURSE_GUIDE.md](COURSE_GUIDE.md).

## Curriculum

- `Day_01` - The Scalar
- `Day_02` - Autograd
- `Day_03` - Tensors
- `Day_04` - Tensor Ops
- `Day_05` - Modules and Neurons
- `Day_06` - Layers and MLPs
- `Day_07` - Loss Functions
- `Day_08` - Optimization
- `Day_09` - Training Loops
- `Day_10` - MNIST

## Included Now

- `Day_01/01_the_scalar_template.ipynb` - teaching notebook with markdown
  intuition and blank code cells
- `Day_01/vuktorch/engine.py` - Day 01 scalar engine before automatic
  traversal
- `Day_01/examples/manual_backprop.py` - worked manual gradient example
- `Day_01/HOMEWORK.md` - Day 01 challenge prompt for `__pow__`
- `Day_02/vuktorch/engine.py` - scalar autograd engine with `.backward()`
- `Day_02/examples/autograd_demo.py` - automatic backprop demo

Each day folder now has a README that states the goal, build target, and
exercise so the course is coherent from end to end.
