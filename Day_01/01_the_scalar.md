# Day 01: The Scalar

**Series:** PyTorch from Scratch: The Zero-to-One Framework

This is the full Day 01 lesson in one markdown file. Open this file, read it top to bottom, and record the video from it. The lesson is broken into small steps, and each step has a short practice task so people can follow along without getting lost in huge code blocks.

## Goal

Build the smallest useful abstraction in the framework: a scalar `Value` object that can live in a computation graph and remember how gradients should flow backward through it.

## What You Are Building

By the end of Day 01, you should understand how to build:

- a `Value` class that wraps a single scalar
- support for `+` and `*`
- a graph of parent nodes via `_prev`
- a local backward rule stored on each result node
- a manual understanding of gradient flow

This is the first-principles version of autograd before we automate anything.

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

## Step 1: Wrap a Float

Start with a scalar object that remembers both the value and the graph history.

```python
class Value:
    def __init__(self, data, _children=(), _op="", label=""):
        self.data = float(data)
        self.grad = 0.0
        self._prev = set(_children)
        self._op = _op
        self.label = label
        self._backward = lambda: None
```

```python
    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"
```

### Try This Yourself

Create a `Value` instance and print it.

### Hint

Use a simple number like `Value(3.14)`.

## Step 2: Add `+`

The first operation is addition. It creates a new node and stores a backward rule on that output node.

```python
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")
```

```python
        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
```

```python
        out._backward = _backward
        return out
```

### Try This Yourself

Ask the learner to implement addition from memory in a fresh file or notebook cell.

### Hint

Each side of addition receives the same gradient.

## Step 3: Add `*`

Multiplication is the second operation. It follows the same structure, but the local derivatives are different.

```python
    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")
```

```python
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
```

```python
        out._backward = _backward
        return out
```

### Try This Yourself

Ask the learner to write the local backward rule for multiplication without looking.

### Hint

Each parent gets the other parent’s value multiplied by the output gradient.

## Step 4: Build a Tiny Graph

We will use the same toy graph throughout Day 01:

```text
d = a * b + c
```

Forward values:

- `a = 2.0`
- `b = -3.0`
- `c = 10.0`
- `e = a * b = -6.0`
- `d = e + c = 4.0`

### Create the values

```python
a = Value(2.0, label="a")
b = Value(-3.0, label="b")
c = Value(10.0, label="c")
```

```python
e = a * b
e.label = "e"
```

```python
d = e + c
d.label = "d"
```

### Inspect the forward pass

```python
print("Forward pass:")
print("a =", a.data)
print("b =", b.data)
print("c =", c.data)
print("e = a * b =", e.data)
print("d = e + c =", d.data)
```

### Try This Yourself

Have the learner predict the outputs before running the code.

### Hint

Work left to right: first multiply, then add.

## Step 5: Backward Pass by Hand

The key idea is local derivatives. Each operation only needs to know how to send gradient to its immediate parents.

For addition, each input gets a gradient of `1`.

For multiplication, the left input gets the right value and the right input gets the left value.

```python
d.grad = 1.0
```

```python
d._backward()
```

```python
e._backward()
```

```python
print("Gradients after manual backprop:")
print("dd/da =", a.grad)
print("dd/db =", b.grad)
print("dd/dc =", c.grad)
print("dd/de =", e.grad)
print("dd/dd =", d.grad)
```

Expected values:

- `dd/da = -3.0`
- `dd/db = 2.0`
- `dd/dc = 1.0`
- `dd/de = 1.0`
- `dd/dd = 1.0`

### Try This Yourself

Ask the learner to explain why `a` gets `-3.0` and `b` gets `2.0`.

### Hint

`d` depends on `e`, and `e = a * b`, so the gradient through `e` gets multiplied by the other input.

## What To Say On Camera

The important point is not the arithmetic. It is the structure:

- every node stores its own value
- every node knows its parents
- every node knows the local gradient rule for the operation that created it

That is enough to build autograd later. Day 02 will automate the traversal, but the local rules stay the same.

## Homework: Add `__pow__`

Extend the scalar engine by adding power for a scalar `Value` and a numeric exponent.

### Task

Implement `x ** n` for a scalar `Value`.

### Forward Rule

```text
x^n
```

### Backward Rule

```text
d(x^n) / dx = n * x^(n-1)
```

### Constraints

- Only support numeric exponents.
- Keep the style consistent with `__add__` and `__mul__`.
- Accumulate gradients with `+=`.
- Do not add tensor support here.

### Starter Stub

Use this as the starting point in `vuktorch/engine.py`:

```python
    def __pow__(self, other):
        # only support scalar numeric powers for now
        out = Value(self.data ** other, (self,), f"**{other}")

        def _backward():
            self.grad += other * (self.data ** (other - 1)) * out.grad

        out._backward = _backward
        return out
```

### Try This Yourself

Add `__pow__` to the engine, then test `x = Value(2.0); y = x ** 3`.

### Hint

The forward pass is `self.data ** other`, and the derivative is `other * self.data ** (other - 1)`.

### Example To Match

When it is implemented, this should work:

```python
x = Value(2.0)
y = x ** 3
y.grad = 1.0
y._backward()

print(y.data)  # 8.0
print(x.grad)  # 12.0
```

### Homework Checks

Your implementation should pass these checks:

```python
x = Value(2.0)
assert (x ** 2).data == 4.0
```

```python
x = Value(3.0)
y = x ** 3
y.grad = 1.0
y._backward()
assert y.data == 27.0
assert x.grad == 27.0
```

### Stretch Goal

If you want an extra challenge, implement division using a negative power once `__pow__` works.

```python
# def __truediv__(self, other):
#     return self * (other ** -1)
```

## End Of Day 01

That is enough for the first lesson. The next day is about turning these local rules into a full `.backward()` traversal.
