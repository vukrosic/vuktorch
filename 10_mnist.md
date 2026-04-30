# Day 10: MNIST

**Series:** PyTorch from Scratch: The Zero-to-One Framework

**Community:** [Become AI Researcher](https://skool.com/become-ai-researcher-2669/about)

## Goal

Run the framework end to end on MNIST and prove that the full stack composes.

## What You Should Understand By The End

- how the earlier abstractions fit together on a real dataset
- what extra work real data introduces
- how training and evaluation look in a complete script
- why end-to-end tests matter for framework design

## Why Day 10 Exists

A framework is not finished when individual pieces work in isolation. It is finished when:

- data can be loaded
- a model can run forward
- a loss can be computed
- gradients can flow
- parameters can update
- evaluation can report progress

MNIST is the first compact dataset in the series that proves the whole stack works together.

## Step 1: Load The Data

The dataset step usually needs:

- image tensors
- label targets
- train and test splits
- a simple batching strategy

The important design idea is that the rest of the training code should not care where the data came from once loading is done.

## Step 2: Define The Model

At this point the model can stay simple:

- flatten the image
- feed it into an MLP
- produce class scores

The point of Day 10 is not to squeeze out benchmark accuracy. The point is to show the framework survives contact with real input shapes and labels.

## Step 3: Train End To End

The same loop structure from Day 09 still applies:

```python
for x_batch, y_batch in train_data:
    optimizer.zero_grad()
    pred = model(x_batch)
    loss = loss_fn(pred, y_batch)
    loss.backward()
    optimizer.step()
```

This is an important milestone. The same ideas from the tiny scalar graph now drive real training code.

## Step 4: Evaluate

After training, compute:

- test loss
- test accuracy

Then inspect a few predictions manually.

Questions:

- which digits are easy?
- which mistakes happen often?
- are errors random, or do certain classes get confused?

## Step 5: What MNIST Proves

If the script trains and produces sensible predictions, then the framework has shown:

- scalar autograd works
- tensor autograd works
- modules compose
- losses work
- optimization works
- training infrastructure works

That is why Day 10 matters. It is the integration test for the whole course.

## Homework

### Homework 1: Inspect Wrong Predictions

After one training run, collect a few mistakes and look at them by hand.

This is good homework because it moves the lesson beyond "the code runs" toward "the model behavior makes sense."

### Homework 2: Change Model Width

Train two different MLP widths and compare:

- training loss
- test accuracy
- speed

This builds intuition for model capacity instead of treating architecture as magic.

### Stretch: Normalize Inputs

Try a simple input normalization step and compare whether training becomes smoother.

## End Of Day 10

Day 10 is complete once the framework can train on a real dataset and produce outputs that make the whole series feel coherent.

Continue with the community here: [Become AI Researcher](https://skool.com/become-ai-researcher-2669/about).
