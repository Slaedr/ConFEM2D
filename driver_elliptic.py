
import sys
import gc
import numpy as np
import numpy.linalg
import scipy.sparse as scs
import scipy.sparse.linalg as scsl
from matplotlib import pyplot as plt
from mesh import *
from fem import *
from output import *

# user input
numberofmeshes = 3
meshfile = "../Meshes-and-geometries/squarehole"
dirBCnum = np.array([2,4])
#dirBCnum = np.array([12,13])
ngauss = 6

# preprocess file names
meshes = []
outs = []
for imesh in range(numberofmeshes):
	meshes.append(meshfile+str(imesh)+".msh")
	outs.append(meshfile+str(imesh)+".vtu")

data = np.zeros((numberofmeshes,3),dtype=np.float64)

for imesh in range(numberofmeshes):
    # mesh
    mio = Mesh2dIO()
    mio.readGmsh(meshes[imesh])
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

    # compute
    #A = np.zeros((m.npoin,m.npoin),dtype=np.float64)
    Ai = []; Aj = []; Av = []
    b = np.zeros(m.npoin, dtype=np.float64)
    assemble(m, dirBCnum, Ai, Aj, Av, b, poly_degree, ngauss)
    #assemble(m, dirBCnum, A, b, poly_degree, ngauss)
    A = scs.csr_matrix((Av,(Ai,Aj)), shape=(m.npoin,m.npoin))
    print("Solving linear system..")
    #x = np.linalg.solve(A, b)
    #x,info = scsl.gmres(A,b,tol=1e-5, maxiter=500)
    lu = scsl.splu(A)
    x = lu.solve(b)
    print("Solved")
    #(Ad,bd,dirflags) = removeDirichletRowsAndColumns(m,A,b,dirBCnum)
    #x,xd = solveAndProcess(m, Ad, bd, dirflags)

    #print("Final relative residual = " + str(np.linalg.norm(np.dot(A,x)-b,2)/np.linalg.norm(b,2)))

    # output
    writePointScalarToVTU(m, outs[imesh], "poisson", x)

    # errors - uncomment for non-exact L2 and H1 error norms
    #err = np.zeros(m.npoin, dtype=np.float64)
    #err[:] = exact_sol(m.coords[:,0], m.coords[:,1])
    ##writePointScalarToVTU(m, "../fem2d-results/"+meshes[imesh]+"-exact.vtu", "exact", err)
    #err[:] = err[:] - x[:]
    #l2norm, h1norm = compute_norm(m, err, poly_degree, ngauss)

    # uncomment for "exact" L2 errors, but no H1 error computation
    l2norm = compute_norm_exact(m, x, poly_degree, ngauss)
    h1norm = l2norm

    print("Mesh " + str(imesh))
    print("  The mesh size paramter, the error's L2 norm and its H1 norm (log base 10):")
    print("  "+str(np.log10(m.h))  + " " + str(np.log10(l2norm)) + " " + str(np.log10(h1norm))+ "\n")
    data[imesh,0] = np.log10(m.h)
    data[imesh,1] = np.log10(l2norm)
    data[imesh,2] = np.log10(h1norm)

    #gc.collect()

# plots
n = numberofmeshes

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


"""plt.title("Grid-refinement (legend: slopes)") # + title)
plt.xlabel("Log mesh size")
plt.ylabel("Log error")
plt.legend()
plt.show()"""
