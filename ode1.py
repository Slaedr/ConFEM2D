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

def factorizeMatrix(A):
    return scsl.splu(A)

class LinearOde1:
    """ Base class for first-order ODE integration."""
    def __init__(self, mesh, dirBCnum):
        self.m = mesh
        self.dbn = dirBCnum
        self.dt = 0.0
        self.dirflags = np.zeros(mesh.npoin,dtype=np.int32)

    def setOperators(self, A, M):
        # Set the spatial operators (eg. stiffnesss matrix) and mass matrix
        pass

    def step(self, un):
        # solve for the change u^{n+1}-u^n
        b = self.A.dot(un)
        applyZeroDirichletRHS(b, self.dirflags)
        deltau = self.fM.solve(b)
        return un+deltau

class LForwardEuler(LinearOde1):
    """ Forward Euler scheme """
    def __init__(self, mesh, dirBCnum, cfl, coeff):
        LinearOde1.__init__(self, mesh, dirBCnum)
        #self.m = mesh
        #self.dbn = dirBCnum
        #self.ftime = final_time
        self.dt = cfl*m.h*m.h/coeff

    def setOperators(self, A, M):
        # NOTE: Modifies M
        M = M.multiply(1.0/dt)
        applyDirichletPenaltiesLHS(self.m, M, self.dirBCnum, self.dirflags)
        self.fM = factorizeMatrix(M)
        self.A = A

class LBackwardEuler(LinearOde1):
    """ Backward Euler scheme """
    def __init__(self, mesh, dirBCnum, dt):
        LinearOde1.__init__(self, mesh, dirBCnum)
        #self.m = mesh
        #self.dbn = dirBCnum
        #self.ftime = final_time
        self.dt = dt

    def setOperators(self, A, M):
        # NOTE: M is modified
        self.A = A
        M = M.multiply(1.0/dt) - A
        # apply penalties to Dirichlet rows and classify points into Dirichlet or not (dirflags)
        applyDirichletPenaltiesLHS(self.m, M, self.dirBCnum, self.dirflags)
        self.fM = factorizeMatrix(M)

class LCrankNicolson(LinearOde1):
    def __init__(self, mesh, dirBCnum, dt):
        LinearOde1.__init__(self, mesh, dirBCnum)
        self.dt = dt

    def setOperators(self, A, M):
        # NOTE: M is modified
        self.A = A
        M = M.multiply(1.0/dt) - A.multiply(0.5)
        # apply penalties to Dirichlet rows and classify points into Dirichlet or not (dirflags)
        applyDirichletPenaltiesLHS(self.m, M, self.dirBCnum, self.dirflags)
        self.fM = factorizeMatrix(M)

