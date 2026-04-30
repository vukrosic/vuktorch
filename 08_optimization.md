# Day 08: Optimization

**Series:** PyTorch from Scratch: The Zero-to-One Framework

**Community:** [Become AI Researcher](https://skool.com/become-ai-researcher-2669/about)

## Goal

Turn gradients into parameter updates by building a simple SGD optimizer.

## What You Should Understand By The End

- why backward alone does not make a model learn
- what an optimizer actually does
- why `zero_grad()` matters
- how one update step can reduce loss

## Why Day 08 Exists

Backpropagation tells you how the loss changes with each parameter. It does not change the parameters for you.

Optimization is the step that closes the loop.

## Step 1: Build SGD

The simplest optimizer stores:

- a list of parameters
- a learning rate

```python
class SGD:
    def __init__(self, params, lr=0.01):
        self.params = list(params)
        self.lr = lr
```

## Step 2: Add `step()`

Each parameter moves opposite its gradient.

```python
def step(self):
    for p in self.params:
        p.data -= self.lr * p.grad
```

This is gradient descent in its most stripped-down form.

Question: why do we subtract the gradient instead of add it?

## Step 3: Add `zero_grad()`

Gradients accumulate by design, so they must be cleared before the next training step.

```python
def zero_grad(self):
    for p in self.params:
        p.grad = 0.0 * p.grad
```

If you forget this step, each batch mixes its gradients with the previous ones.

## Step 4: Try One Manual Update

```python
pred = model(x)
loss = mse(pred, y)

optimizer.zero_grad()
loss.backward()
optimizer.step()
```

That four-line sequence is the core of almost every training loop in the course.

## Step 5: Verify That Loss Can Decrease

On a tiny example, run one step and compare loss before and after.

Question: if the loss increases after one step, what are the first two things to check?

Good answers:

- learning rate might be too high
- gradients or backward rules might be wrong

## Homework

### Homework 1: One-Step Loss Check

Pick a tiny model and dataset, run one update step, and verify that the loss is lower afterward.

This is good homework because it tests the whole training path:

- forward
- loss
- backward
- update

### Homework 2: Add Weight Decay Intuition

Without implementing it fully yet, write down how the update rule would change if parameters should also be nudged toward zero.

### Stretch: Momentum

Think about what extra state the optimizer would need if each step should remember part of the previous update.

## End Of Day 08

Day 08 is complete once gradients stop being diagnostic information and start becoming actual learning updates.

Continue with the community here: [Become AI Researcher](https://skool.com/become-ai-researcher-2669/about).
