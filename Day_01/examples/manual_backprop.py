from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vuktorch.engine import Value


def main():
    a = Value(2.0, label="a")
    b = Value(-3.0, label="b")
    c = Value(10.0, label="c")

    e = a * b
    e.label = "e"
    d = e + c
    d.label = "d"

    print("Forward pass:")
    print(f"a = {a.data}")
    print(f"b = {b.data}")
    print(f"c = {c.data}")
    print(f"e = a * b = {e.data}")
    print(f"d = e + c = {d.data}")

    # Day 01 does not automate traversal yet.
    # We seed the output gradient by hand, then call local backward rules in
    # the correct reverse order.
    d.grad = 1.0
    d._backward()
    e._backward()

    print("\nGradients after manual backprop:")
    print(f"dd/da = {a.grad}")
    print(f"dd/db = {b.grad}")
    print(f"dd/dc = {c.grad}")
    print(f"dd/de = {e.grad}")
    print(f"dd/dd = {d.grad}")


if __name__ == "__main__":
    main()
