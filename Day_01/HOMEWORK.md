# Homework Task - Add `__pow__`

Extend the scalar `Value` class so it can raise a value to a constant power.

## Goal

Implement:

```python
def __pow__(self, other):
    ...
```

Assume `other` is a Python `int` or `float`, not another `Value`.

## Forward Pass

If:

```python
x = Value(3.0)
y = x ** 2
```

then `y.data` should be `9.0`.

## Backward Pass

Use the derivative:

```text
d(x^n)/dx = n * x^(n - 1)
```

That means when gradients flow backward through:

```python
y = x ** n
```

you should accumulate:

```python
x.grad += n * (x.data ** (n - 1)) * y.grad
```

## Constraints

- Only support scalar exponents for now.
- Keep the implementation consistent with the existing `__add__` and `__mul__`
  style.
- Make sure gradients accumulate with `+=`, not `=`.

## Test Yourself

After implementing it, this should work:

```python
x = Value(2.0)
y = x ** 3
y.backward()

print(y.data)  # 8.0
print(x.grad)  # 12.0
```

## Stretch Goal

Use your new `__pow__` to implement division like this:

```python
def __truediv__(self, other):
    return self * (other ** -1)
```
