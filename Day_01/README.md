# Day 01 - The Scalar

Day 01 is now packaged as one markdown file so you can open a single file,
read it top to bottom, and record the lesson without switching between lesson
and homework files.

## Main File

- `01_the_scalar.md` - the full Day 01 lesson plus the homework section at the
  end

## Supporting Files

- `examples/manual_backprop.py` - a tiny worked example showing gradients by
  hand
- `vuktorch/engine.py` - the Day 01 scalar engine used by the examples

## What The Lesson Covers

The markdown lesson walks through:

1. why a scalar needs to remember graph history
2. how `Value` stores `data`, `grad`, `_prev`, `_op`, and `_backward`
3. a worked example for `d = a * b + c`
4. manual backpropagation, step by step
5. a built-in homework section for adding `__pow__`

## Quick Demo

Run the worked example if you want a separate terminal demo:

```bash
python3 Day_01/examples/manual_backprop.py
```
