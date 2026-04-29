# Day 01: The Scalar

**Series:** PyTorch from Scratch: The Zero-to-One Framework

This is the full Day 01 recording script in one file. Open this markdown file, read it top to bottom, and record from it directly.

## Goal

Build the smallest useful object in the whole framework: a scalar `Value` that stores a number, remembers where it came from, and knows how gradients should flow backward.

## Why This Exists

A plain Python float can hold a number like `3.0`, but it cannot answer:

- where did this number come from?
- what operation created it?
- which earlier values should receive gradient from it?

That is the gap the `Value` class fills. Day 01 is not about speed or abstraction. It is about making autograd feel inevitable.

## What The Scalar Needs To Store

Our scalar node needs five things:

- `data` for the actual number
- `grad` for the gradient accumulated at that node
- `_prev` for the parent nodes
- `_op` for the operation that produced the node
- `_backward` for the local chain-rule step

If those five things exist, then the rest of autograd is mostly graph traversal.

## Step 1: Start With The Constructor

Start with the minimum state. At this point the class does not know how to add or multiply yet. It only knows how to represent one node.

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

The important line here is `self._prev = set(_children)`. That is what turns a plain number into a graph node. Every result can now remember which earlier values created it.

Also notice `self.grad = 0.0`. Gradients will accumulate into that field later when we manually backpropagate.

## Step 2: Make The Object Printable

Once you start creating nodes, you need a fast way to inspect them.

```python
    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"
```

This is small, but it matters for teaching. If the object prints clearly, the learner can see both the forward value and the gradient state at every step.

### Quick Check

Ask the learner to run this:

```python
x = Value(3.14)
print(x)
```

Expected idea: the number shows up in `data`, and `grad` is still `0.0` because no backward pass has happened yet.

## Step 3: Add Addition

Now we create the first real operation: `+`.

```python
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")
```

The first line normalizes inputs so `Value(2.0) + 3.0` still works. The second line creates a brand new node called `out`. That new node stores:

- the new scalar value
- both parents
- the symbol `"+"` so we know how it was created

At this point the graph structure already exists. We still need the local derivative rule.

```python
        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
```

This closure is the whole Day 01 idea in miniature. The output node knows how to send gradient into its parents. For addition, both sides receive `1 * out.grad`.

```python
        out._backward = _backward
        return out
```

The local rule is attached to the result node itself. That means every result carries its own tiny backward instruction.

### Quick Check

Ask the learner to explain this sentence in plain English: "the output of `x + y` sends the same gradient to both parents."

### Hint

The derivative of `x + y` with respect to `x` is `1`, and with respect to `y` is also `1`.

## Step 4: Add Multiplication

Multiplication uses the same pattern. The structure stays the same. Only the local derivative rule changes.

```python
    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")
```

Again, a new output node is created. It stores the product, the parents, and the operation symbol.

```python
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
```

This is the product rule in local form. The left parent receives the right parent's value times the output gradient. The right parent receives the left parent's value times the output gradient.

```python
        out._backward = _backward
        return out
```

Same pattern as addition. New node, local backward rule, return the result.

### Quick Check

Ask the learner what gradient should flow into `a` and `b` if `e = a * b`.

### Hint

`de/da = b` and `de/db = a`.

## Step 5: Put The Whole Day 01 Class Together

At this point the full Day 01 class is still very small.

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

That is enough to build a graph and manually push gradients through it.

## Step 6: Build A Tiny Graph

We will use one simple graph for the whole lesson:

```text
a = 2.0
b = -3.0
c = 10.0
e = a * b
d = e + c
```

This is small enough to compute by hand, which is exactly what we want.

```python
a = Value(2.0, label="a")
b = Value(-3.0, label="b")
c = Value(10.0, label="c")
```

These are the leaf nodes. They were not created by any earlier operation.

```python
e = a * b
e.label = "e"
```

This creates the first intermediate node. `e` now remembers that it came from `a` and `b` through multiplication.

```python
d = e + c
d.label = "d"
```

This creates the final output. `d` remembers that it came from `e` and `c` through addition.

## Step 7: Inspect The Forward Pass

Before talking about gradients, make the forward numbers concrete.

```python
print("Forward pass:")
print("a =", a.data)
print("b =", b.data)
print("c =", c.data)
print("e = a * b =", e.data)
print("d = e + c =", d.data)
```

Expected values:

- `e = -6.0`
- `d = 4.0`

### Quick Check

Ask the learner to predict `e` and `d` before running the code.

### Hint

Multiply first, then add.

## Step 8: Seed The Output Gradient

Backward propagation always starts by saying "the output changes with respect to itself by 1."

```python
d.grad = 1.0
```

This is the seed of the whole backward pass. Without this line, no gradient has anywhere to start from.

## Step 9: Run The Local Backward Rules Manually

Day 01 does not implement automatic graph traversal yet. We call the local rules ourselves in reverse order.

```python
d._backward()
```

This pushes gradient from `d` into `e` and `c`.

```python
e._backward()
```

This pushes the gradient that reached `e` back into `a` and `b`.

That is manual backpropagation. The rules were always there. We just executed them ourselves.

## Step 10: Inspect The Gradients

Now print the gradient values and explain what each one means.

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

Why these numbers:

- `dd/dc = 1` because `d = e + c`
- `dd/de = 1` because `d = e + c`
- `dd/da = b = -3` because `e = a * b`
- `dd/db = a = 2` because `e = a * b`

### Quick Check

Ask the learner why `a.grad` becomes negative while `b.grad` becomes positive.

### Hint

The sign comes from the other multiplicand.

## What The Learner Should Understand Before Moving On

If the learner understands these three sentences, Day 01 worked:

- every result node stores its parents
- every result node stores a local backward rule
- backpropagation is just the repeated application of those local rules in reverse order

Day 02 will automate the reverse traversal, but it will not change the local derivative logic from today.

## Homework

The homework should be short, directly tied to the lesson, and testable with plain Python. The first version needed one fix: it should not rely on `.backward()` yet, because Day 01 has not implemented automatic traversal. The homework below stays consistent with the lesson.

### Homework 1: Add `__pow__`

Implement `x ** n` for a scalar `Value`, where `n` is a Python number.

What to implement:

- forward rule: `x ** n`
- backward rule: `n * x ** (n - 1)`

Constraints:

- only support numeric exponents
- keep the style consistent with `__add__` and `__mul__`
- accumulate gradients with `+=`

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

Why this is good homework:

- it reuses the same pattern from `+` and `*`
- it forces the learner to separate forward logic from backward logic
- it stays scalar-only, so the difficulty does not jump too early

Test it like this:

```python
x = Value(2.0)
y = x ** 3
y.grad = 1.0
y._backward()

assert y.data == 8.0
assert x.grad == 12.0
```

### Homework 2: Support Right-Hand Numbers

Make expressions like `2 + Value(3.0)` and `2 * Value(3.0)` work.

Why this is good homework:

- it is small
- it improves ergonomics immediately
- it teaches that Python dispatch has left-hand and right-hand variants

Hint:

- implement `__radd__`
- implement `__rmul__`

Expected shape:

```python
    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other
```

Quick test:

```python
x = Value(3.0)
assert (2 + x).data == 5.0
assert (2 * x).data == 6.0
```

### Stretch Homework: Add Division

Only do this after `__pow__` works.

```python
    def __truediv__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return self * (other ** -1)
```

This is a good stretch task because it teaches code reuse. Division does not need new math if power already exists.

## End Of Day 01

Day 01 is done when the learner can read a tiny graph, explain where each gradient comes from, and implement one new scalar operation by following the same local-rule pattern.
