
import sys
import numpy as np
import numpy.linalg
from mesh import *
from fem import *
from output import *

# user input
meshfile = "../Meshes-and-geometries/squarehole-coarse.msh"
dirBCnum = np.array([2,4])
#dirBCnum = np.array([12,13])
outputfile = "../fem2d-results/squarehole-coarse.vtu"

# mesh
mio = Mesh2dIO()
mio.readGmsh(meshfile)
m = Mesh2d(mio.npoin, mio.nelem, mio.nbface, mio.maxnnodel, mio.maxnnofa, mio.nbtags, mio.ndtags, mio.coords, mio.inpoel, mio.bface, mio.nnodel, mio.nfael, mio.nnofa, mio.dtags)
mio = 0

# compute
A = np.zeros((m.npoin,m.npoin),dtype=np.float64)
b = np.zeros(m.npoin, dtype=np.float64)
assemble(m, dirBCnum, A, b)
x = np.linalg.solve(A, b)

# output
writePointScalarToVTU(m, outputfile, "poisson", x)

# errors
err = np.zeros(m.npoin, dtype=np.float64)
err[:] = exact_sol(m.coords[:,0], m.coords[:,1])
writePointScalarToVTU(m, "../fem2d-results/exact.vtu", "exact", err)
err[:] = err[:] - x[:]
l2norm, h1norm = compute_norm(m, err)
print("The mesh size paramter, the error's L2 norm and its H1 norm (log base 10):")
print(str(np.log10(m.h)) + " " + str(np.log10(l2norm)) + " " + str(np.log10(h1norm))+"\n")

