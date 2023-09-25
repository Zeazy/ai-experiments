

## Notes

Tensors in pytorch can be thought of as generalizations of scalers, vectors and matrices:
1. Scalars are 0-dimensional tensors
2. Vectors are 1-dimensional tensors
3. Matrices are 2-dimensional tensors

`tensor.shape` returns the shape of the tensor. For scalars, this is an empty tuple. For vectors, this is a tuple with one element. For matrices, this is a tuple with two elements.

`torch.ones(2,2)` creates a 2x2 matrix of ones. `torch.zeros(2,2)` creates a 2x2 matrix of zeros. `torch.rand(2,2)` creates a 2x2 matrix of random numbers between 0 and 1.

