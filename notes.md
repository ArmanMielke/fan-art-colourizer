# Notes

[Info on reproducibility with PyTorch](https://pytorch.org/docs/stable/notes/randomness.html)

## Regarding the use of inplace operations

- This [pytorch autograd doc](https://pytorch.org/docs/master/notes/autograd.html#in-place-operations-on-variables) discourages the use of inplace operations
- This [thread](https://discuss.pytorch.org/t/guidelines-for-when-and-why-one-should-set-inplace-true/50923) says one should use inplace operations unless there's an error, in order to save memory
- I will follow the advice from the thread for this project and use inplace operations for ReLU
