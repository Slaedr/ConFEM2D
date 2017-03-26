
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

np.set_printoptions(linewidth=200, threshold=5000)

# ----------
# user input
# ----------
finaltime = 0.2
dt = 0.01
ntimesteps = 3
meshfile = "../Meshes-and-geometries/discquad4"
dirBCnum = np.array([2,])
ngauss = 6
a = 1.0
#-----------
############

def rhs_func(x,y):
    return 0.0

def stiffness_coeff_func(x,y):
    # equal to -a!
    return -1.0

def mass_coeff_func(x,y):
    return 1.0

def dirichlet_function(x,y):
    return 0.0

class ExactSol:
    def __init__(self, a):
        self.r2 = jn_zeros(0,2)[-1]
        self.a = a
        print("ExactSol: Bessel zero = "+str(self.r2))

    def eval(self,x,y,t):
        return np.exp(-self.r2*self.r2*self.a*self.a*t)*j0(self.r2*np.sqrt(x*x+y*y))

funcs = CoeffFunctions(rhs_func, stiffness_coeff_func, mass_coeff_func, dirichlet_function, a)
exactsol = ExactSol(a)

# preprocess file names
outs = []
basename = meshfile.split('/')[-1]
for it in range(ntimesteps):
    outs.append("../fem2d-results/"+basename+"_t-FDF1"+str(it)+".vtu")

data = np.zeros((ntimesteps,2),dtype=np.float64)
    
mio = Mesh2dIO()
mio.readGmsh(meshfile+".msh")
m = Mesh2d(mio.npoin, mio.nelem, mio.nbface, mio.maxnnodel, mio.maxnnofa, mio.nbtags, mio.ndtags, 
        mio.coords, mio.inpoel, mio.bface, mio.nnodel, mio.nfael, mio.nnofa, mio.dtags)
mio = 0

poly_degree = 0
if m.nnodel[0] == 3:
    poly_degree = 1
elif m.nnodel[0] == 6:
    poly_degree = 2
print("Approximation polynomial degree = " + str(poly_degree))

# assemble
Ac = COOMatrix(m.npoin, m.npoin)
Mc = COOMatrix(m.npoin, m.npoin)
b = np.zeros(m.npoin, dtype=np.float64)
assemble_stiffness(m, Ac, poly_degree, ngauss, stiffness_coeff_func)
assemble_mass(m, Mc, poly_degree, ngauss, mass_coeff_func)

uexact = np.zeros(m.npoin)
uexact[:] = exactsol.eval(m.coords[:,0], m.coords[:,1], finaltime)


for it in range(ntimesteps):
    print("Time step " + str(it))
    A = scs.csc_matrix((Ac.vals,(Ac.rind,Ac.cind)), shape=(m.npoin,m.npoin))
    M = scs.csc_matrix((Mc.vals,(Mc.rind,Mc.cind)), shape=(m.npoin,m.npoin))
    #be = LForwardEuler(m, dirBCnum, dt)
    #be = LBackwardEuler(m, dirBCnum, dt)
    be = LCrankNicolson(m, dirBCnum, dt)
    be.setOperators(A,M)

    # set initial solution
    un = np.zeros(m.npoin)
    un[:] = exactsol.eval(m.coords[:,0], m.coords[:,1], 0.0)

    t = 0.0; step = 0
    while t < finaltime - 1e-10:
        be.step(un)
        t += dt
        step += 1
        if step % 20 == 0:
            print("    Time step " + str(step) + ": Time = " + str(t))
    print("  Final time = " + str(t))
    
    l2norm = compute_error(m, un, poly_degree, ngauss, finaltime, exactsol)
    print("Time-step " + str(it))
    print("  The mesh size paramter, the error's L2 norm and its H1 norm (log base 10):")
    print("  "+str(np.log10(dt))  + " " + str(np.log10(l2norm)) + "\n")
    data[it,0] = np.log10(dt)
    data[it,1] = np.log10(l2norm)

    writePointScalarToVTU(m, outs[it], "heat", un)
    dt = dt/2.0

# plots
n = ntimesteps

pslope = np.zeros(data.shape[1])
labels = ['L2: ','H1: ', 'Inf:']
symbs = ['o-', 's-', '^-']

for j in range(1,data.shape[1]):
	psigy = data[:,j].sum()
	sigx = data[:,0].sum()
	sigx2 = (data[:,0]*data[:,0]).sum()
	psigxy = (data[:,j]*data[:,0]).sum()

	pslope[j] = (n*psigxy-sigx*psigy)/(n*sigx2-sigx**2)
	print("Slope is " + str(pslope[j]))
	plt.plot(data[:,0],data[:,j],symbs[j-1],label=labels[j-1]+str(pslope[j]))


plt.title("Temporal convergence (legend: slopes)") # + title)
plt.xlabel("Log time-step")
plt.ylabel("Log error")
plt.legend()
plt.show()
