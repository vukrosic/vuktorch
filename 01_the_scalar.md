# Day 01: The Scalar

**Series:** PyTorch from Scratch: The Zero-to-One Framework

## Goal

Build a scalar `Value` object that stores a number, remembers which values created it, and knows how to send gradients backward.

## What You Should Understand By The End

- why a plain float is not enough for autograd
- what information each scalar node must store
- how `+` and `*` create graph nodes
- how local backward rules produce gradients

## Why A Scalar Class Exists

A Python float can hold `3.0`, but it cannot answer:

- where did this value come from?
- which earlier values produced it?
- what gradients should flow into those earlier values?

That is what the `Value` class adds.

## What A Node Stores

Each scalar node needs:

- `data` for the number
- `grad` for the gradient at that node
- `_prev` for the parent nodes
- `_op` for the operation that created the node
- `_backward` for the local chain-rule step

If those pieces exist, then autograd becomes a graph traversal problem.

## Step 1: Create The Node

Start with the smallest useful version of the class.

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

`self._prev` is what makes this a graph node instead of just a wrapper around a float. `self.grad` starts at `0.0` and will be filled during backpropagation.

## Step 2: Make It Easy To Inspect

```python
    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"
```

This makes the object readable while debugging.

Question: what should `print(Value(3.14))` show before any backward pass runs?

## Step 3: Add Addition

Addition creates a new output node.

```python
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")
```

The result stores the new value, both parents, and the operation symbol.

Now attach the local backward rule.

```python
        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
```

For addition, both parents receive the same gradient.

```python
        out._backward = _backward
        return out
```

Question: why do both parents receive `1 * out.grad`?

Hint: `d(x + y)/dx = 1` and `d(x + y)/dy = 1`.

## Step 4: Add Multiplication

Multiplication uses the same structure, but a different derivative rule.

```python
    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")
```

The node creation logic is unchanged. Only the local derivative rule changes.

```python
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
```

Each parent receives the other parent's value multiplied by the output gradient.

```python
        out._backward = _backward
        return out
```

Question: if `e = a * b`, what are `de/da` and `de/db`?

Hint: `de/da = b` and `de/db = a`.

## Full Day 01 Class

```python
class Value:
    def __init__(self, data, _children=(), _op="", label=""):
        self.data = float(data)
        self.grad = 0.0
        self._prev = set(_children)
        self._op = _op
        self.label = label
        self._backward = lambda: None

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad

        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out
```

That is enough for a tiny computation graph.

## Step 5: Build A Tiny Graph

Use this graph for the whole lesson:

```text
a = 2.0
b = -3.0
c = 10.0
e = a * b
d = e + c
```

Create the nodes:

```python
a = Value(2.0, label="a")
b = Value(-3.0, label="b")
c = Value(10.0, label="c")

e = a * b
e.label = "e"

d = e + c
d.label = "d"
```

Forward pass:

```python
print("e =", e.data)
print("d =", d.data)
```

Expected values:

- `e = -6.0`
- `d = 4.0`

Question: what happens if you multiply first and then add?

## Step 6: Run Backward By Hand

Day 01 does not automate graph traversal yet. The backward rules already exist, so call them manually in reverse order.

Start by seeding the output gradient:

```python
d.grad = 1.0
```

Then run the local backward rules:

```python
d._backward()
e._backward()
```

Print the gradients:

```python
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

Why these values:

- `dd/dc = 1` because `d = e + c`
- `dd/de = 1` because `d = e + c`
- `dd/da = b = -3` because `e = a * b`
- `dd/db = a = 2` because `e = a * b`

Question: why is `a.grad` negative while `b.grad` is positive?

## Homework

These tasks stay close to the lesson and can be tested with plain Python.

### Homework 1: Add `__pow__`

Implement `x ** n` for a scalar `Value`, where `n` is a Python number.

Rules:

- forward: `x ** n`
- backward: `n * x ** (n - 1)`

Starter shape:

```python
    def __pow__(self, other):
        assert isinstance(other, (int, float))
        out = Value(self.data ** other, (self,), f"**{other}")

        def _backward():
            self.grad += other * (self.data ** (other - 1)) * out.grad

        out._backward = _backward
        return out
```

Test:

```python
x = Value(2.0)
y = x ** 3
y.grad = 1.0
y._backward()

assert y.data == 8.0
assert x.grad == 12.0
```

This is good Day 1 homework because it reuses the same pattern as `+` and `*`.

### Homework 2: Support Right-Hand Numbers

Make these work:

```python
2 + Value(3.0)
2 * Value(3.0)
```

Implementation shape:

```python
    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other
```

Test:

```python
x = Value(3.0)
assert (2 + x).data == 5.0
assert (2 * x).data == 6.0
```

### Stretch: Add Division

Only do this after `__pow__` works.

```python
    def __truediv__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return self * (other ** -1)
```

## End Of Day 01

Day 01 is complete once you can explain where each gradient comes from and implement one new scalar operation using the same local-rule pattern.
