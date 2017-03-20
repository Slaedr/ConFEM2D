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

class Ode1:
    """ Base class for first-order ODE integration."""
    def __init__(self, mesh, dirBCnum, final_time):
        self.m = mesh
        self.dbn = dirBCnum
        self.ftime = final_time
        self.dt = 0.0

    def assembleLHS(self):
        pass

    def solve(self):
        pass

class ForwardEuler(Ode1):
    """ Forward Euler scheme"""
    def __init__(self, mesh, dirBCnum, final_time, cfl, coeff):
        self.m = mesh
        self.dbn = dirBCnum
        self.ftime = final_time
        self.dt = cfl*m.h*m.h/coeff
