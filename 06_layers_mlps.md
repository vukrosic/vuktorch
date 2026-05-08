# Day 06: Sequential Models and MLPs

**Series:** PyTorch from Scratch: The Zero-to-One Framework

**Community:** [Become AI Researcher](https://skool.com/become-ai-researcher-2669/about)

## Goal

Compose `Linear` layers and activation modules into a small multilayer
perceptron.

## What You Should Understand By The End

- how modules compose into larger modules
- why activations are modules too
- how `Sequential` makes model definitions readable
- how an `MLP` turns architecture sizes into layers

## Why Day 06 Exists

One linear layer is useful, but most interesting models stack several
transformations:

```text
input -> linear -> activation -> linear -> activation -> linear -> output
```

The math does not change. The design challenge is making the composition clean.

## Step 1: Add Activation Modules

Activation functions already exist on `Tensor`, but model code is cleaner when
they can also be used as layers.

```python
class ReLU(Module):
    def forward(self, x):
        return x.relu()

class Tanh(Module):
    def forward(self, x):
        return x.tanh()
```

These modules do not own parameters. They still fit the same interface because
they transform tensors during the forward pass.

Question: why is it useful for parameter-free layers and parameterized layers to
share the same module interface?

## Step 2: Build Sequential

`Sequential` stores layers in order and calls them one after another.

```python
class Sequential(Module):
    def __init__(self, *layers):
        self.layers = list(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
```

This is simple, but powerful. The list can contain any module: `Linear`, `ReLU`,
`Tanh`, or another custom module.

## Step 3: Define An MLP

An MLP is a sequence of linear layers with activations between hidden layers.

```python
class MLP(Sequential):
    def __init__(self, in_features, hidden, out_features, activation="tanh"):
        ...
```

For example:

```python
model = MLP(2, [8, 8], 2)
```

This means:

- input size: `2`
- hidden layer 1: `2 -> 8`
- hidden layer 2: `8 -> 8`
- output layer: `8 -> 2`

The final layer does not need an activation because classification logits should
stay as raw scores before cross-entropy.

## Step 4: Choose The Activation

The final implementation supports `"tanh"` and `"relu"`:

```python
activation_cls = Tanh if activation == "tanh" else ReLU
```

That keeps the interface small while still letting you compare two common
nonlinearities.

Question: what would happen if every layer were linear and there were no
activation functions?

## Step 5: Test A Tiny Model

```python
x = Tensor([[0.5, -1.0]])
model = MLP(2, [4, 4], 2)
logits = model(x)
```

Now inspect:

```python
print(logits.shape)
print(len(model.parameters()))
```

Expected ideas:

- `logits.shape` should be `(1, 2)`
- parameters come from the nested `Linear` layers
- activation modules do not add parameters

## Where This Lives In Code

The final project implementation for this lesson lives in `vuktorch/nn.py`.

Look for:

- `ReLU` and `Tanh` for activation modules
- `Sequential` for ordered composition
- `MLP` for turning an architecture specification into a complete model
- `Module.parameters` for recursively collecting parameters from nested layers

The model is used directly in `examples/mlp_moons.py`, where `MLP(2, [8, 8], 2)`
trains as a classifier. It is also used in `tests/test_nn.py` to prove that the
module stack can learn.

## Homework

### Homework 1: Change The Width

Try:

```python
MLP(2, [4], 2)
MLP(2, [16, 16], 2)
```

Compare parameter counts and training behavior.

### Homework 2: Add `__repr__`

Make `Sequential` or `MLP` print a compact architecture summary.

This is useful because portfolio reviewers should be able to understand the
model at a glance.

### Stretch: Validate Activation Names

Raise a clear error if someone passes an activation name other than `"tanh"` or
`"relu"`.

## End Of Day 06

Day 06 is complete once a deeper model feels like a clean composition of reusable
modules.

Continue with the community here: [Become AI Researcher](https://skool.com/become-ai-researcher-2669/about).
