
import sys
import gc
import numba
import numpy as np
import scipy.sparse as scs
import scipy.sparse.linalg as scsl
from scipy.special import jn_zeros, jn
from matplotlib import pyplot as plt
from mesh import *
from matrices import COOMatrix
from fem import *
from ode1 import *
from output import *

# user input
numberofmeshes = 3
meshfile = "../Meshes-and-geometries/disc"
dirBCnum = np.array([2,])
#dirBCnum = np.array([12,13])
ngauss = 6

#@jit(nopython=True)
def rhs_func(x,y):
    return 0.0

#@jit(nopython=True)
def stiffness_coeff_func(x,y):
    return 1.0

#@jit(nopython=True)
def mass_coeff_func(x,y):
    return 1.0

#@jit(nopython=True)
def exact_sol(x,y,t):
    return np.exp(t)

#@jit(nopython=True)
def dirichlet_function(x,y):
    return 0.0
