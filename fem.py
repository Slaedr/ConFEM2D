""" @brief Assembly routines of local and global FE matrices.
"""

import numpy as np
from numba import jit
from mesh import *
from quadrature import GLQuadrature1D, GLQuadrature2DTriangle

@jit(nopython=True, cache=True)
def coeff_mass(x, y):
    return x + 2.0*y

@jit(nopython=True, cache=True)
def coeff_stiffness(x, y):
    return x*x + y*y - x*y

@jit(nopython=True, cache=True)
def localMassMatrix(m, ielem, localmass):
    """ Computes the local mass matrix of element ielem in mesh m.
    The output array localmass needs to be pre-allocated."""
    pass

@jit(nopython=True, cache=True)
def localStiffnessMatrix(m, ielem, localstif):
    """ Computes the local stiffness matrix of element ielem in mesh m.
    The output array localmass needs to be pre-allocated."""
    pass

def localLoadVector_domain(m, ielem, localload):
    """ Computes the domain integral part of the local load vector.
    localload must be pre-allocated.
    """
    pass

def localLoadVector_boundary(m, iface, localload):
    """ Computes the local boundary integral part of load vector for Neumann BCs.
    localload must be allocated before passing to this.
    """
    pass
