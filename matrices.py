"""
@brief Defines some matrix-related functionality.
"""

import numpy as np

class COOMatrix:
    def __init__(self, nrows, ncols):
        self.rind = []
        self.cind = []
        self.vals = []
        self.m = nrows
        self.n = ncols


#@jit(nopython=True, cache=True)
def matvec(A,b):
    # Matvec
    x = np.zeros(b.shape)
    for i in range(A.shape[0]):
        for j in range(b.shape[0]):
            x[i] += A[i,j]*b[j]
    return x

#@jit(nopython=True, cache=True)
def dotprod(a,b):
    # Scalar product
    x = 0.0
    for i in range(a.shape[0]):
        x += a[i]*b[i]
    return x
