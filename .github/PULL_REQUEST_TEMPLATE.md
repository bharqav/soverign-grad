## Summary

Brief description of what this PR changes.

## Type of Change

- [ ] Bug fix (non-breaking)
- [ ] New operator / layer
- [ ] Performance optimization
- [ ] Documentation update
- [ ] Build / CI change
- [ ] Breaking change

## Changes Made

- 

## Testing

Describe how you tested this change:

- [ ] Added test case to `src/test_suite.cpp`
- [ ] Added `gradcheck` for new differentiable operator
- [ ] Ran full test suite locally: `./build/sovereign_grad_test`
- [ ] Ran sanitizer build: `cmake -DENABLE_SANITIZERS=ON ... && ./build/sovereign_grad_test`

## Numerical Correctness

If adding a differentiable operator, paste the `gradcheck` output:

```
[gradcheck] max_abs_err: X.XXXX, max_rel_err: X.XXXX → PASSED
```

## Checklist

- [ ] Forward pass is implemented and tested
- [ ] Backward pass is implemented if the operator should be differentiable
- [ ] No unused includes, debug prints, or `assert(false)` leftover
- [ ] Code compiles without warnings on GCC with `-Wall -Wextra`
- [ ] `CONTRIBUTING.md` workflow followed
