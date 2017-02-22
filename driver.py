
import sys
import numpy as np
import numpy.linalg
from mesh import *
from fem import *

# user input
meshfile = " "
dirBCnum = 2
outputfile = " "

# mesh
mio = Mesh2dIO()
mio.readGmsh(meshfile)
m = Mesh2d(mio.npoin, mio.nelem, mio.nbface, mio.maxnnodel, mio.nbtags, mio.ndtags, mio.coords, mio.inpoel, mio.bface, mio.nnodel, mio.nnofa, mio.dtags)

# compute
A = np.zeros((m.npoin,m.npoin),dtype=np.float64)
b = np.zeros(m.npoin, dtype=np.float64)
assemble(m, dirBCnum, A, b)
x = np.linalg.solve(A, b)

# output
writePointScalarToVTU(m, outputfile, "poisson", x)
