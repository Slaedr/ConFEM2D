"""
@brief Integration of first-order ODEs (in time)
"""

import numpy as np
from numpy import sin, cos, arctan
import scipy.sparse as scs
import scipy.sparse.linalg as scsl
from mesh import *
from quadrature import GLQuadrature1D, GLQuadrature2DTriangle
from elements import *
from fem import *

np.set_printoptions(linewidth=200)

def factorMatrix(A):
    return scsl.splu(A)

class LinearOde1:
    """ Base class for first-order ODE integration."""
    def __init__(self, mesh, dirBCnum, final_time):
        self.m = mesh
        self.dbn = dirBCnum
        self.ftime = final_time
        self.dt = 0.0

    def setOperators(self, A, M):
        # Set the spatial operators (eg. stiffnesss matrix) and mass matrix
        pass

    def step(self, un):
        # Code for executing 1 time step goes here
        pass

class LForwardEuler(LinearOde1):
    """ Forward Euler scheme """
    def __init__(self, mesh, dirBCnum, final_time, cfl, coeff):
        self.m = mesh
        self.dbn = dirBCnum
        self.ftime = final_time
        self.dt = cfl*m.h*m.h/coeff

    def setOperators(self, A, M):
        # NOTE: A is modified!
        self.fM = factorMatrix(M)
        self.A = A.multiply(dt)

    def step(self, un):
        # 1 step of forward Euler
        b = self.A.dot(un)
        deltau = self.fM.solve(b)
        return un+deltau

class LBackwardEuler(LinearOde1):
    pass

class LCrankNicolson(LinearOde1):
    pass
