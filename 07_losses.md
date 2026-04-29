# Day 07: Loss Functions

**Series:** PyTorch from Scratch: The Zero-to-One Framework

## Goal

Turn model outputs into a scalar training signal by implementing loss functions.

## What You Should Understand By The End

- why training needs a scalar objective
- how MSE works for regression
- why cross-entropy is common for classification
- why losses usually reduce across a batch

## Why Day 07 Exists

Without a loss, gradients do not point anywhere useful. A loss converts "model output" into "how wrong the model is."

That wrongness must become a scalar, because backward needs one output to propagate from.

## Step 1: Add Mean Squared Error

For predictions `pred` and targets `target`:

```python
loss = ((pred - target) ** 2).mean()
```

This is a good first loss because every piece is already familiar:

- subtraction
- power
- mean

Question: why is the square useful here instead of plain subtraction?

## Step 2: Think About Batch Reduction

A batch usually produces one loss per example. Training usually wants one scalar for the batch.

That is why reduction matters:

- `sum()` keeps scale dependent on batch size
- `mean()` keeps scale more stable as batch size changes

For a first framework, `mean()` is the cleaner default.

## Step 3: Add Cross-Entropy Intuition

Cross-entropy is the standard choice for classification because it strongly rewards confident correct predictions and strongly penalizes confident wrong ones.

For logits `x`, the conceptual form is:

```text
cross_entropy = -log(probability assigned to the correct class)
```

In practice, this usually means:

- convert logits to probabilities with softmax
- select the correct class probability
- take negative log
- reduce across the batch

## Step 4: Compare The Two Losses

MSE asks: "how far are the numbers from the target numbers?"

Cross-entropy asks: "how much probability mass did the model assign to the right class?"

That is why MSE is a natural default for regression, while cross-entropy is the better default for classification.

## Step 5: Try A Tiny Example

Regression-style example:

```python
pred = Tensor([2.5, 0.0, 2.0, 8.0])
target = Tensor([3.0, -0.5, 2.0, 7.0])
loss = ((pred - target) ** 2).mean()
```

Classification-style question:

- if the correct class gets low probability, should the loss be small or large?

## Homework

### Homework 1: Write `mse(pred, target)`

Wrap the MSE formula in a reusable function.

Test:

```python
pred = Tensor([1.0, 2.0])
target = Tensor([1.0, 4.0])
loss = mse(pred, target)
```

Expected value:

- `((0.0 ** 2) + ((-2.0) ** 2)) / 2 = 2.0`

### Homework 2: Implement A Simple Cross-Entropy

Start with the single-example case before worrying about batches.

That keeps the shape logic manageable and lets you focus on the meaning of the loss first.

### Stretch: Numerical Stability

Look into why large logits can make naive softmax or log-softmax implementations unstable.

## End Of Day 07

Day 07 is complete once model outputs can be turned into one scalar number that clearly says how wrong the model is.
