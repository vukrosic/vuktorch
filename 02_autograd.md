# Day 02: Autograd

**Series:** PyTorch from Scratch: The Zero-to-One Framework

**Community:** [Become AI Researcher](https://skool.com/become-ai-researcher-2669/about)

## Goal

Turn manual backpropagation into a real `.backward()` method on `Value`.

## What You Should Understand By The End

- why local derivative rules are not enough by themselves
- what topological order means in a computation graph
- why backward must run in reverse topological order
- how `.backward()` automates the Day 01 process

## Why Day 02 Exists

Day 01 already had the right math. Each node knew its local derivative rule. The problem was execution: backward only worked if you manually called each node in the right reverse order.

Day 02 solves exactly that problem. The chain rule stays the same. The new idea is graph traversal.

## The Core Idea

For a graph like this:

```text
a = 2.0
b = -3.0
c = 10.0
e = a * b
d = e + c
```

`d` depends on `e`, and `e` depends on `a` and `b`. That means backward must visit `d` before `e`, and `e` before `a` and `b`.

That ordering is the whole point of topological sorting.

## Step 1: Add A Graph Traversal Helper

The job of `build_topo` is to collect all nodes that influence the current output and arrange them so parents appear before children.

```python
def build_topo(v):
    topo = []
    visited = set()

    def build(node):
        if node not in visited:
            visited.add(node)
            for child in node._prev:
                build(child)
            topo.append(node)

    build(v)
    return topo
```

This uses depth-first search. It walks to the leaves first, then appends nodes on the way back up. That means the resulting list is forward order, from inputs toward the output.

Question: why is it useful that children are visited before the node that depends on them?

## Step 2: Add `.backward()`

Once the graph is collected, backward becomes straightforward.

```python
def backward(self):
    topo = build_topo(self)
    self.grad = 1.0

    for node in reversed(topo):
        node._backward()
```

The line `self.grad = 1.0` seeds the output gradient. The reverse loop then applies each local backward rule in the only order that makes sense.

This is still the same chain rule from Day 01. The difference is that the graph is now traversed automatically.

## Step 3: Put It Into `Value`

```python
class Value:
    ...

    def backward(self):
        topo = []
        visited = set()

        def build(node):
            if node not in visited:
                visited.add(node)
                for child in node._prev:
                    build(child)
                topo.append(node)

        build(self)
        self.grad = 1.0

        for node in reversed(topo):
            node._backward()
```

Keeping `build` inside `.backward()` is fine for now. It keeps the idea local and readable.

## Step 4: Check It On The Day 01 Graph

```python
a = Value(2.0, label="a")
b = Value(-3.0, label="b")
c = Value(10.0, label="c")

e = a * b
d = e + c

d.backward()
```

Now print the gradients:

```python
print(a.grad)
print(b.grad)
print(c.grad)
print(e.grad)
print(d.grad)
```

Expected values:

- `a.grad = -3.0`
- `b.grad = 2.0`
- `c.grad = 1.0`
- `e.grad = 1.0`
- `d.grad = 1.0`

Question: why should these match the manual Day 01 result exactly?

## What Changed Conceptually

Three important things did not change:

- each node still stores a local backward rule
- each rule still only knows about its immediate parents
- gradients still accumulate with `+=`

Only one thing changed: traversal is now automatic.

## Homework

### Homework 1: Reuse A Node Twice

Test a graph where one value is used more than once.

```python
a = Value(3.0)
b = a + a
b.backward()

assert b.data == 6.0
assert a.grad == 2.0
```

This is good homework because it proves that gradients must accumulate, not overwrite.

### Homework 2: Add `__sub__`

Implement subtraction using the operations you already have.

One clean version is:

```python
def __neg__(self):
    return self * -1

def __sub__(self, other):
    return self + (-other)
```

Test:

```python
a = Value(5.0)
b = Value(2.0)
c = a - b
c.backward()

assert c.data == 3.0
assert a.grad == 1.0
assert b.grad == -1.0
```

### Stretch: Clear Old Gradients

Run backward twice on the same graph and observe what happens. Then add a helper that resets gradients to zero before reuse.

## End Of Day 02

Day 02 is complete once `.backward()` produces the same gradients that Day 01 required you to compute by hand.

Continue with the community here: [Become AI Researcher](https://skool.com/become-ai-researcher-2669/about).
