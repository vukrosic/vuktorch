from vuktorch import Tensor


def main():
    x = Tensor(2.0, requires_grad=True)
    y = Tensor(-3.0, requires_grad=True)
    z = (x * y + x**2).tanh()
    z.backward()

    print("z =", z.item())
    print("dz/dx =", x.grad)
    print("dz/dy =", y.grad)


if __name__ == "__main__":
    main()
