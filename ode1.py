"""
@brief Integration of first-order ODEs (in time)
"""

import numpy as np
from numpy import sin, cos, arctan
import scipy.sparse as scs
import scipy.sparse.linalg as scsl
import scipy.linalg as scl
from mesh import *
from quadrature import GLQuadrature1D, GLQuadrature2DTriangle
from elements import *
from fem import *

np.set_printoptions(linewidth=200)

def factorizeMatrix(A):
    return scsl.splu(A)

class LinearOde1Delta:
    """ Base class for first-order ODE integration """
    def __init__(self, mesh, dirBCnum):
        self.m = mesh
        self.dbn = dirBCnum
        self.dt = 0.0
        self.dirflags = np.zeros(mesh.npoin,dtype=np.int32)

    def setOperators(self, A, M):
        # Set the spatial operators (eg. stiffnesss matrix) and mass matrix
        pass

    def step(self, un):
        # solve for the change u^{n+1} - u^n
        b = self.A.dot(un)
        applyDirichletRHS(b, self.dirflags, 0.0)
        deltau = self.fM.solve(b)
        un[:] = un[:] + deltau[:]

class LForwardEuler(LinearOde1Delta):
    """ Forward Euler scheme """
    def __init__(self, mesh, dirBCnum, dt):
        LinearOde1Delta.__init__(self, mesh, dirBCnum)
        self.dt = dt

    def setOperators(self, A, M):
        # NOTE: Modifies M
        M = M.multiply(1.0/self.dt)
        applyDirichletPenaltiesLHS(self.m, M, self.dbn, self.dirflags)
        self.fM = factorizeMatrix(M)
        self.A = A

class LBackwardEuler(LinearOde1Delta):
    """ Backward Euler scheme """
    def __init__(self, mesh, dirBCnum, dt):
        LinearOde1Delta.__init__(self, mesh, dirBCnum)
        self.dt = dt

    def setOperators(self, A, M):
        # NOTE: M is modified
        self.A = A
        M = M.multiply(1.0/self.dt)-A
        applyDirichletPenaltiesLHS(self.m, M, self.dbn, self.dirflags)
        self.fM = factorizeMatrix(M)

class LCrankNicolson(LinearOde1Delta):
    def __init__(self, mesh, dirBCnum, dt):
        LinearOde1Delta.__init__(self, mesh, dirBCnum)
        self.dt = dt

    def setOperators(self, A, M):
        # NOTE: M is modified
        self.A = A
        M = M.multiply(1.0/self.dt) - A.multiply(0.5)
        # apply penalties to Dirichlet rows and classify points into Dirichlet or not (dirflags)
        applyDirichletPenaltiesLHS(self.m, M, self.dbn, self.dirflags)
        self.fM = factorizeMatrix(M)

class LBDF1:
    """ Alternative backward Euler formulation"""
    def __init__(self,mesh, dirBCnum, dirvalue, dt):
        self.m = mesh
        self.dbn = dirBCnum
        self.dirval = dirvalue
        self.dt = dt
        self.dirflags = np.zeros(mesh.npoin,dtype=np.int32)

    def setOperators(self, A, M):
        self.resop = M.multiply(1.0/self.dt)
        temp = self.resop - A
        applyDirichletPenaltiesLHS(self.m, temp, self.dbn, self.dirflags)
        self.jac = factorizeMatrix(temp)

    def step(self, un):
        # Overwrites the input
        b = self.resop.dot(un)
        applyDirichletRHS(b, self.dirflags, self.dirval)
        un[:] = self.jac.solve(b)

