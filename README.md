ConFEM 2D
=========

Implementation of 2D finite elements written in Python. Performance is achieved using the Numpy and Scipy libraries and Numba JIT compilation.

Finite Elements
---------------
Geometric mappings (from reference elements) are Lagrange only. Currently, P1 and P2 triangular elements are implemented, but any other P is easy to add.
The basis functions are currently Lagrange too.

Spatial operators
-----------------
Currently, the diffusion operator (stiffness matrix) and identity operator (mass matrix) are implemented.

Time-stepping
-------------
Forward Euler, backward Euler and Crank-Nicolson schemes are available.
