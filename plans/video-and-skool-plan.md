# Tiny PyTorch From Scratch: AI Research Portfolio Project

This is the production plan for the public YouTube course and the paid Skool continuation.

## Main Title

Tiny PyTorch From Scratch: AI Research Portfolio Project

## YouTube Promise

By the end of the 10 daily videos, viewers have a working tiny PyTorch-style framework they can put on GitHub and explain in AI/ML interviews.

## 10-Day Video Plan

| Day | Video | What To Tell People | Code Artifact | Portfolio Checkpoint |
|----:|-------|---------------------|---------------|----------------------|
| 1 | The Scalar | Neural networks are built from tiny differentiable operations. Start with one number that remembers how it was created. | `Tensor` scalar operations | explain what a computation graph is |
| 2 | Autograd | Backprop is not magic; it is reverse traversal of a graph with local gradients. | `backward()` topological sort | answer why reverse-mode autodiff fits neural nets |
| 3 | Tensors | Real models need arrays, shapes, and broadcasting. | NumPy-backed `Tensor` | explain scalar vs tensor autodiff |
| 4 | Tensor Ops | A framework is mostly correct gradients for common operations. | matmul, reductions, activations | explain matmul gradients |
| 5 | Modules And Parameters | Models are reusable objects that own trainable tensors. | `Module`, `Parameter`, `Linear` | explain what a parameter is |
| 6 | MLPs | Stack layers into a real neural network. | `Sequential`, `MLP`, activations | draw the model architecture |
| 7 | Losses | Training needs one scalar objective. | MSE and cross entropy | explain stable cross entropy |
| 8 | Optimizers | Learning is parameter updates driven by gradients. | SGD and Adam | explain optimizer state |
| 9 | Training Loops | Put forward, loss, backward, and step into one loop. | two-moons classifier example | show loss/accuracy improving |
| 10 | Portfolio Project | Clean the repo, tests, README, and extension path. | tests, examples, portfolio docs | publish a polished GitHub repo |

## Day 11 Full Course

Publish the full masterclass as one long video after the 10 daily lessons.

Suggested full-course title:

Tiny PyTorch From Scratch: Full AI Research Portfolio Project

## What Goes To Skool

Skool should not duplicate the public course. Skool turns the public project into interview proof.

### Skool Assets

- README review checklist
- interview answer worksheet
- resume bullet worksheet
- project extension menu
- gradient debugging checklist
- member project feedback thread
- final portfolio submission checklist

### Skool Weekly/End Challenge

Each member must add one unique extension:

- new operation
- new loss
- gradient checker
- model save/load
- benchmark against PyTorch
- tiny transformer block
- visualization of the computation graph

### Skool Feedback Prompts

- Post your GitHub repo.
- Post one screenshot or terminal output proving training works.
- Explain broadcasting gradients in your own words.
- Pick one extension and describe why it makes your project different.
- Write one resume bullet and one interview answer.
