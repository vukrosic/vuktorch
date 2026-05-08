# Day 10: End-to-End Classifier

**Series:** PyTorch from Scratch: The Zero-to-One Framework

**Community:** [Become AI Researcher](https://skool.com/become-ai-researcher-2669/about)

## Goal

Run the framework end to end on a real classification-shaped problem and prove
that the full stack composes.

## What You Should Understand By The End

- how the earlier abstractions fit together in one script
- how data, model, loss, optimizer, and metrics connect
- why an end-to-end demo matters for portfolio credibility
- how to explain the project as a complete ML engineering system

## Why Day 10 Exists

A framework is not finished when individual pieces work in isolation. It is
finished when:

- data can be loaded or generated
- a model can run forward
- a loss can be computed
- gradients can flow
- parameters can update
- evaluation can report progress

The project demo uses a two-moons classifier because it is small, visual, fast,
and does not require downloading a dataset. That makes it a good final proof for
a portfolio repo.

## Step 1: Create A Small Dataset

The demo creates a synthetic two-class dataset:

```python
x, y = make_moons()
```

The input has two features per point, and the target is a class label:

- `x.shape == (n, 2)`
- `y.shape == (n,)`

This is enough to test a real classifier without adding dataset dependencies.

## Step 2: Define The Model

The classifier is a small MLP:

```python
model = MLP(2, [8, 8], 2)
```

That architecture means:

- two input features
- two hidden layers with width 8
- two output logits, one per class

The final output is logits, not probabilities. `cross_entropy` handles the
softmax-style probability logic inside the loss.

## Step 3: Train End To End

The same loop structure from Day 09 now trains the full model:

```python
logits = model(Tensor(x))
loss = cross_entropy(logits, y)

optim.zero_grad()
loss.backward()
optim.step()
```

This is the whole stack in motion:

- `Tensor` stores data and gradients
- `MLP` produces logits
- `cross_entropy` creates a scalar training objective
- `.backward()` propagates gradients
- `SGD.step()` updates parameters

## Step 4: Track Accuracy

The demo logs both loss and accuracy:

```python
preds = logits.data.argmax(axis=1)
acc = (preds == y).mean()
```

Loss tells you whether optimization is improving the objective. Accuracy tells
you whether the classifier is making better decisions.

Question: why might loss improve before accuracy changes much?

## Step 5: What The Demo Proves

If `examples/mlp_moons.py` trains and reports sensible accuracy, then the
framework has shown:

- scalar and tensor autograd work
- broadcasting and matrix multiplication gradients work
- modules compose into a real model
- cross-entropy produces useful gradients
- an optimizer can update trainable parameters
- the training loop can improve a model

That is the portfolio story. You did not just implement isolated operations. You
built enough of a deep learning framework to train a classifier.

## Where This Lives In Code

The final project demo for this lesson lives in `examples/mlp_moons.py`.

Look for:

- `make_moons` for the dataset
- `MLP(2, [8, 8], 2)` for the classifier
- `SGD(model.parameters(), lr=0.02)` for optimization
- `cross_entropy(logits, y)` for the loss
- printed `loss` and `acc` values for the demo output

Supporting implementation lives across the package:

- `vuktorch/tensor.py` for autograd
- `vuktorch/nn.py` for modules and MLPs
- `vuktorch/losses.py` for cross-entropy
- `vuktorch/optim.py` for SGD

## Homework

### Homework 1: Change Model Width

Train two different MLP widths and compare:

- training loss
- accuracy
- speed

This builds intuition for model capacity instead of treating architecture as
magic.

### Homework 2: Try A Different Learning Rate

Compare `lr=0.005`, `lr=0.02`, and `lr=0.1`.

Watch for:

- slow learning
- stable improvement
- unstable loss

### Stretch: Add MNIST

MNIST is a strong extension after the framework is stable. To add it cleanly,
you would need:

- dataset loading
- image flattening or reshaping
- train and test splits
- batching
- final accuracy reporting

That makes MNIST a good portfolio extension rather than a dependency required
for the core course.

## End Of Day 10

Day 10 is complete once the framework can train a classifier end to end and you
can explain how every part of the stack contributes.

Continue with the community here: [Become AI Researcher](https://skool.com/become-ai-researcher-2669/about).
