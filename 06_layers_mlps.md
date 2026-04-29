# Day 06: Layers and MLPs

**Series:** PyTorch from Scratch: The Zero-to-One Framework

## Goal

Compose neurons into layers and layers into a small multilayer perceptron.

## What You Should Understand By The End

- how a layer groups neurons
- how an MLP stacks layers
- why composition matters more than inventing new math here
- how parameter collection scales from one module to many

## Why Day 06 Exists

A single neuron is useful for understanding, but not for real modeling. The next step is to combine many neurons into reusable blocks.

Day 06 is mostly about composition. The math engine and neuron logic already exist.

## Step 1: Build A Layer

A layer is just a list of neurons with the same input size.

```python
class Layer(Module):
    def __init__(self, nin, nout):
        self.neurons = [Neuron(nin) for _ in range(nout)]
```

Each neuron receives the same input vector, but produces its own scalar output.

## Step 2: Write The Layer Forward Pass

```python
def __call__(self, x):
    outs = [n(x) for n in self.neurons]
    return outs[0] if len(outs) == 1 else outs
```

This keeps the API convenient:

- one output neuron returns one scalar-like result
- multiple output neurons return a list of results

Question: why might returning a bare single value be nicer than returning a one-element list?

## Step 3: Expose Layer Parameters

```python
def parameters(self):
    return [p for n in self.neurons for p in n.parameters()]
```

This is the first nested parameter collection step.

## Step 4: Build The MLP

An MLP is just a stack of layers.

```python
class MLP(Module):
    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i + 1]) for i in range(len(nouts))]
```

If `nin = 3` and `nouts = [4, 4, 1]`, then the model structure is:

- layer 1: `3 -> 4`
- layer 2: `4 -> 4`
- layer 3: `4 -> 1`

## Step 5: Write The MLP Forward Pass

```python
def __call__(self, x):
    for layer in self.layers:
        x = layer(x)
    return x
```

That is the whole composition story. Each layer transforms the representation and passes it to the next one.

## Step 6: Test A Tiny Model

```python
model = MLP(3, [4, 4, 1])
x = [Tensor(2.0), Tensor(3.0), Tensor(-1.0)]
y = model(x)
```

Now inspect:

```python
print(y)
print(len(model.parameters()))
```

Question: where do all of the parameters in an MLP actually live?

## Homework

### Homework 1: Add Layer `__repr__`

Make a layer print how many neurons it contains.

### Homework 2: Add MLP `__repr__`

Print the full architecture in a compact way, such as the input size and output sizes.

These are good exercises because model inspection becomes important fast once the object graph grows.

### Stretch: Final-Layer Nonlinearity

Most MLPs do not want ReLU on the final output. Adjust the design so hidden layers can be nonlinear while the final layer can stay linear.

## End Of Day 06

Day 06 is complete once a deeper model feels like simple composition rather than a new kind of abstraction.
