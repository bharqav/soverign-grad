---
name: Feature Request / New Operator
about: Propose a new operator, layer, or capability
title: '[Feature] '
labels: enhancement
assignees: ''
---

## What would you like added?

Describe the operator, layer, or capability you'd like to see.

## Motivation

Why is this useful? Which model architectures or workflows require it?

## Proposed API

```cpp
// Example C++ API sketch (can be rough)
Tensor result = some_op(input, param);
```

## Gradient derivation

If adding a differentiable operator, sketch the backward pass:

```
d_input = ...
```

## Alternatives considered

Have you considered workarounds or alternative approaches?
