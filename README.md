ConFEM 2D
=========

Implementation of 2D finite element using continuous trial and test spaces, written in Python. Performance is achieved using the Numpy and Scipy libraries and Numba JIT compilation.

Finite Elements
---------------
Geometric mappings (from reference elements) are Lagrange only. Currently, P1 and P2 triangular elements are implemented, but any other P is easy to add.
The basis functions are currently Lagrange too.
