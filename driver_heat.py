
import sys
import gc
import numba
import numpy as np
import scipy.sparse as scs
import scipy.sparse.linalg as scsl
from scipy.special import jn_zeros, j0
from matplotlib import pyplot as plt
from mesh import *
from matrices import COOMatrix
from fem import *
from ode1 import *
from output import *

# user input
finaltime = 1.0
dt = 0.1
numberofmeshes = 1
meshfile = "../Meshes-and-geometries/disc"
dirBCnum = np.array([2,])
ngauss = 6
a = 1.0

def rhs_func(x,y):
    return 0.0

def stiffness_coeff_func(x,y):
    # equal to a!
    return 1.0

def mass_coeff_func(x,y):
    return 1.0

def dirichlet_function(x,y):
    return 0.0

class ExactSol:
    def __init__(self, a):
        self.r2 = jn_zeros(0,2)[-1]
        self.a = a
        print("ExactSol: Bessel zero = "+str(r2))

    def eval(x,y,t):
        return np.exp(-self.r2*self.r2*self.a*self.a*t)*j0(self.r2*np.sqrt(x*x+y*y))

funcs = CoeffFunctions(rhs_func, stiffness_coeff_func, mass_coeff_func, dirichlet_function, a)
exactsol = ExactSol(a)

# preprocess file names
meshes = []
outs = []
basename = fname.split('/')[-1]
for imesh in range(numberofmeshes):
	meshes.append(meshfile+str(imesh)+".msh")
	outs.append("../fem2d-results/"+basename+str(imesh)+".vtu")

data = np.zeros((numberofmeshes,3),dtype=np.float64)

for imesh in range(numberofmeshes):
    mio = Mesh2dIO()
    mio.readGmsh(meshes[imesh])
    mio.readGmsh(meshfile)
    m = Mesh2d(mio.npoin, mio.nelem, mio.nbface, mio.maxnnodel, mio.maxnnofa, mio.nbtags, mio.ndtags, 
            mio.coords, mio.inpoel, mio.bface, mio.nnodel, mio.nfael, mio.nnofa, mio.dtags)
    mio = 0

    poly_degree = 0
    if m.nnodel[0] == 3:
        poly_degree = 1
    elif m.nnodel[0] == 6:
        poly_degree = 2
    if imesh == 0:
        print("Approximation polynomial degree = " + str(poly_degree))

    # assemble
    Ac = COOMatrix(m.npoin, m.npoin)
    Mc = COOMatrix(m.npoin, m.npoin)
    b = np.zeros(m.npoin, dtype=np.float64)
    assemble_stiffness(m, dirBCnum, Ac, b, poly_degree, ngauss, stiffness_coeff_func)
    assemble_mass(m, dirBCnum, Mc, b, poly_degree, ngauss, mass_coeff_func)
    A = scs.csc_matrix((Ac.vals,(Ac.rind,Ac.cind)), shape=(m.npoin,m.npoin))
    M = scs.csc_matrix((Mc.vals,(Mc.rind,Mc.cind)), shape=(m.npoin,m.npoin))

    be = LBackwardEuler(m, dirBCnum, dt)
    be.setOperators(A,M)

    # set initial solution
    un = np.zeros(m.npoin)
    un[:] = exactsol.eval(m.coords(:,0), m.coords(:,1), 0.0)
    writePointScalarToVTU(m, outs[imesh]+"initial", "heat-initial", un)

    t = 0.0
    while t < finaltime:
        unext = be.step(un)
        un[:] = unext[:]
        t += dt
    print("Final time = " + str(t))

    writePointScalarToVTU(m, outs[imesh], "heat-initial", unext)
