# Day 09: Training Loops

**Series:** PyTorch from Scratch: The Zero-to-One Framework

## Goal

Assemble the model, loss, optimizer, and data into a reusable training loop.

## What You Should Understand By The End

- what happens in one training step
- what happens in one epoch
- why evaluation should be separated from training
- what metrics are worth logging

## Why Day 09 Exists

By now all the pieces exist, but they are still scattered. A training loop is the glue that repeats:

- forward pass
- loss computation
- backward pass
- parameter update

until the model improves.

## Step 1: Define One Training Step

At the smallest useful level, training looks like this:

```python
optimizer.zero_grad()
pred = model(x)
loss = loss_fn(pred, y)
loss.backward()
optimizer.step()
```

Everything else in a training loop is mostly repetition and bookkeeping.

## Step 2: Wrap It In An Epoch

An epoch means iterating over all batches once.

```python
for x_batch, y_batch in data:
    optimizer.zero_grad()
    pred = model(x_batch)
    loss = loss_fn(pred, y_batch)
    loss.backward()
    optimizer.step()
```

If batching is not implemented yet, use one example at a time. The structure is still the same.

## Step 3: Add Evaluation

Training and evaluation should be separate in the code even if they both call the model.

Training updates parameters.

Evaluation only measures performance:

```python
pred = model(x)
loss = loss_fn(pred, y)
```

Question: why is it useful to keep evaluation logic separate even in a toy framework?

## Step 4: Log Something Useful

At minimum, log:

- training loss
- evaluation loss
- accuracy for classification tasks

Logging matters because training is iterative. Without it, you cannot tell whether the model is improving or just running.

## Step 5: Try A Tiny Dataset

A classic small target is XOR or another synthetic dataset where:

- the dataset is tiny
- the training loop runs fast
- you can see loss move within a short run

That makes debugging much easier than jumping straight to a larger dataset.

## Homework

### Homework 1: Train On XOR

Build a tiny dataset and train a small MLP on it.

Track the loss for several epochs and check whether it trends downward.

### Homework 2: Log Accuracy

If the task is classification, add an accuracy metric next to loss.

This is good homework because it forces the loop to produce information a human can actually read.

### Stretch: Early Stopping

Think about a rule that would stop training if validation loss stopped improving.

## End Of Day 09

Day 09 is complete once the framework can run the same training recipe repeatedly instead of only one hand-written update step at a time.
