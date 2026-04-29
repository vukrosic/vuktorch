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

    d.backward()

    print("forward:", d.data)
    print("a.grad:", a.grad)
    print("b.grad:", b.grad)
    print("c.grad:", c.grad)


if __name__ == "__main__":
    main()
