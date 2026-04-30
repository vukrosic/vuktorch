# Day 05: Modules and Neurons

**Series:** PyTorch from Scratch: The Zero-to-One Framework

**Community:** [Become AI Researcher](https://skool.com/become-ai-researcher-2669/about)

## Goal

Introduce a `Module` base class and build a single neuron on top of the tensor engine.

## What You Should Understand By The End

- why models need structure beyond raw tensor ops
- what a `Module` should expose
- how parameters are collected
- how a neuron combines weights, inputs, bias, and nonlinearity

## Why Day 05 Exists

By Day 04, the math engine is strong enough to express a neuron. The next problem is organization. You need a clean way to group parameters and forward logic together.

That is what a `Module` does.

## Step 1: Define A Base Module

Start with the interface, not the complexity.

```python
class Module:
    def parameters(self):
        return []

    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0.0 * p.grad
```

`parameters()` gives the optimizer something to work with. `zero_grad()` gives training a clean reset point.

Question: why should every learned component expose its parameters the same way?

## Step 2: Build A Neuron

A neuron needs:

- one weight per input feature
- one bias
- one forward computation

```python
class Neuron(Module):
    def __init__(self, nin):
        self.w = [Tensor(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Tensor(0.0)
```

This is the first time parameters become persistent model state instead of temporary tensors.

## Step 3: Write The Forward Pass

The neuron output is a weighted sum plus bias.

```python
def __call__(self, x):
    act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
    out = act.relu()
    return out
```

This keeps the neuron readable:

- multiply each input by its weight
- add them up
- add bias
- apply nonlinearity

Question: why is the bias included as the starting value to `sum(...)`?

## Step 4: Expose Parameters

```python
def parameters(self):
    return self.w + [self.b]
```

This is what allows optimizers and training loops to treat the neuron generically.

## Step 5: Test One Neuron

```python
n = Neuron(3)
x = [Tensor(2.0), Tensor(3.0), Tensor(-1.0)]
y = n(x)
```

Now inspect:

```python
print(y)
print(n.parameters())
```

Questions:

- how many parameters should a 3-input neuron have?
- why are weights and bias both parameters?

## Homework

### Homework 1: Add A Linear Neuron Option

Allow the neuron to skip ReLU.

One clean interface:

```python
Neuron(nin, nonlin=True)
```

This is good homework because it introduces small configurability without changing the whole class design.

### Homework 2: Add `__repr__`

Make a neuron print something readable, such as how many inputs it has and whether it uses a nonlinearity.

### Stretch: Nested Modules

Think ahead: if a layer contains many neurons, how should `parameters()` combine them?

## End Of Day 05

Day 05 is complete once a learned component feels like both a function and a container of trainable state.

Continue with the community here: [Become AI Researcher](https://skool.com/become-ai-researcher-2669/about).
