# Day 05: Modules and Linear Layers

**Series:** PyTorch from Scratch: The Zero-to-One Framework

**Community:** [Become AI Researcher](https://skool.com/become-ai-researcher-2669/about)

## Goal

Move from raw tensor operations to reusable model components by implementing
`Parameter`, `Module`, and `Linear`.

## What You Should Understand By The End

- why learned weights should be represented as parameters
- what a `Module` should expose
- how parameters are collected from nested objects
- how a vectorized linear layer replaces hand-written neurons

## Why Day 05 Exists

By Day 04, the tensor engine can do the math needed for a dense layer:

- matrix multiplication
- bias broadcasting
- gradient propagation through both

The next problem is organization. A model should not be a loose pile of tensors.
It should be an object that owns parameters and knows how to run a forward pass.

That is what `Module` gives us.

## Step 1: Create A Parameter

A `Parameter` is a `Tensor` that requires gradients by default.

```python
class Parameter(Tensor):
    def __init__(self, data):
        super().__init__(data, requires_grad=True)
```

This small class matters because it tells the rest of the framework, "this value
is trainable model state."

Question: why should weights require gradients, but input data usually should not?

## Step 2: Define A Module Interface

A `Module` needs two basic behaviors:

- collect trainable parameters
- clear their gradients

```python
class Module:
    def parameters(self):
        ...

    def zero_grad(self):
        for parameter in self.parameters():
            parameter.zero_grad()
```

The final implementation also makes modules callable:

```python
def __call__(self, *args, **kwargs):
    return self.forward(*args, **kwargs)
```

That lets a model feel like a function while still holding state.

## Step 3: Collect Parameters Recursively

Real models contain other modules. A sequential model contains layers. A layer
contains weights and maybe a bias.

That means `parameters()` should search through:

- direct `Parameter` attributes
- child `Module` attributes
- lists or tuples containing parameters or modules

Question: why is recursive parameter collection useful once models get deeper?

## Step 4: Build A Linear Layer

A linear layer computes:

```python
out = x @ weight + bias
```

The layer owns:

- `weight` with shape `(in_features, out_features)`
- optional `bias` with shape `(out_features,)`

```python
class Linear(Module):
    def __init__(self, in_features, out_features, bias=True):
        scale = np.sqrt(2.0 / in_features)
        self.weight = Parameter(np.random.randn(in_features, out_features) * scale)
        self.bias = Parameter(np.zeros(out_features)) if bias else None
```

This is the vectorized version of many neurons at once. Instead of creating one
neuron object per output, matrix multiplication computes all outputs together.

## Step 5: Write The Forward Pass

```python
def forward(self, x):
    out = x @ self.weight
    if self.bias is not None:
        out = out + self.bias
    return out
```

The hard work here depends on earlier lessons:

- Day 04 matmul sends gradients into both `x` and `weight`
- Day 04 broadcasting sends the batch bias gradient back into `bias`

Question: if `x` has shape `(32, 4)` and `weight` has shape `(4, 3)`, what shape
does the output have?

## Step 6: Test One Layer

```python
x = Tensor([[1.0, 2.0, 3.0]])
layer = Linear(3, 2)
y = layer(x)
```

Now inspect:

```python
print(y.shape)
print(len(layer.parameters()))
```

Expected ideas:

- `y.shape` should be `(1, 2)`
- the layer should expose the weight and bias as parameters

## Where This Lives In Code

The final project implementation for this lesson lives in `vuktorch/nn.py`.

Look for:

- `Parameter` for trainable tensors
- `Module.parameters` for recursive parameter discovery
- `Module.zero_grad` for clearing parameter gradients
- `Module.__call__` for forwarding calls into `forward`
- `Linear` for the first reusable neural-network layer

The matching behavior is used in `tests/test_nn.py`, where a small linear model
learns a simple relationship. That is the portfolio proof that the module system
connects correctly to autograd and optimization.

## Homework

### Homework 1: Inspect Parameter Shapes

Create a `Linear(4, 3)` layer and print each parameter shape.

Expected shapes:

- weight: `(4, 3)`
- bias: `(3,)`

### Homework 2: Add A Readable `__repr__`

Make `Linear(4, 3)` print something compact, such as:

```text
Linear(in_features=4, out_features=3, bias=True)
```

This is useful portfolio polish because readable models are easier to debug and
easier to explain.

### Stretch: Add A Bias-Free Layer

Test `Linear(4, 3, bias=False)` and confirm that it exposes only one parameter.

## End Of Day 05

Day 05 is complete once learned layers feel like reusable objects instead of
manual tensor expressions.

Continue with the community here: [Become AI Researcher](https://skool.com/become-ai-researcher-2669/about).
