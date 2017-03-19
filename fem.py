""" @brief Assembly routines of local and global FE matrices.
"""

import numpy as np
import numpy.linalg
from numpy import sin, cos, arctan
from numba import jit, jitclass, int32, float64
from mesh import *
from quadrature import GLQuadrature1D, GLQuadrature2DTriangle
from elements import *

np.set_printoptions(linewidth=200)

#@jit(nopython=True)
def rhs_func(x,y):
    return x*x + y*y-14.0

#@jit(nopython=True)
def stiffness_coeff_func(x,y):
    return 1.0

#@jit(nopython=True)
def mass_coeff_func(x,y):
    return 1.0

#@jit(nopython=True)
def exact_sol(x,y):
    return x*x + y*y - 10.0

#@jit(nopython=True)
def dirichlet_function(x,y):
    return exact_sol(x,y)

"""# for heat equation
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
    return 0.0"""


#@jit(nopython=True, cache=True)
def localStiffnessMatrix(gmap, elem, quadrature, localstiff):
    """ Computes the local stiffness matrix (of size ndofpvarel x ndofpvarel) of element elem.
	ndofpvarel = number of DOFs per variable per element.
    The output array localstiff needs to be pre-allocated with correct dimensions."""

    ndof = localstiff.shape[0]
    localstiff[:,:] = 0.0
    basisg = np.zeros((ndof,2), dtype=np.float64)
    jac = np.zeros((2,2), dtype=np.float64)
    jacinv = np.zeros((2,2), dtype=np.float64)

    for ig in range(quadrature.ng):
        # get quadrature points and weights
        x = quadrature.gp[ig,0]; y = quadrature.gp[ig,1]
        w = quadrature.gw[ig]

        # get basis gradients and jacobian determinant
        elem.getBasisGradients(x,y,basisg)
        jdet = gmap.getJacobian(x,y,jac,jacinv)

        # physical location of quadrature point for coefficient function evaluation
        gx,gy = gmap.evalGeomMapping(x,y)

        # add contribution of this quadrature point to each integral
        for i in range(ndof):
            for j in range(ndof):
                localstiff[i,j] += w * stiffness_coeff_func(gx,gy) * np.dot( np.dot(jacinv.T, basisg[i,:]), np.dot(jacinv.T, basisg[j,:]) ) * jdet


#@jit(nopython=True, cache=True)
def localH1Seminorm2(gmap, elem, quadrature, uvals):
    """ Computes the local H^1 semi-norm squared of the FE function given by uvals on element elem.
    """

    localseminorm2 = 0.0
    ndof = uvals.shape[0]
    basisg = np.zeros((ndof,2), dtype=np.float64)
    jac = np.zeros((2,2), dtype=np.float64)
    jacinv = np.zeros((2,2), dtype=np.float64)

    for ig in range(quadrature.ng):
        # get quadrature points and weights
        x = quadrature.gp[ig,0]; y = quadrature.gp[ig,1]
        w = quadrature.gw[ig]

        # get basis gradients and jacobian determinant
        elem.getBasisGradients(x,y,basisg)
        jdet = gmap.getJacobian(x,y,jac,jacinv)

        # physical location of quadrature point for coefficient function evaluation
        #gx,gy = elem.evalGeomMapping(x,y)

        # add contribution of this quadrature point to the integral
        dofsum1 = np.array([0.0,0.0]); dofsum2 = np.array([0.0, 0.0])
        for i in range(ndof):
            dofsum1[:] += uvals[i] * np.matmul(jacinv.T, basisg[i,:])
            dofsum2[:] += uvals[i] * np.matmul(jacinv.T, basisg[i,:])
        localseminorm2 += np.dot(dofsum1,dofsum2) * w * jdet

    return localseminorm2

#@jit(nopython=True, cache=True)
def localMassMatrix(gmap, elem, quadrature, localmass):
    """ Computes the local mass matrix of element elem.
	quadrature is the 2D quadrature contect to be used; has to be setup beforehand.
    The output array localmass needs to be pre-allocated."""

    ndof = localmass.shape[0]
    localmass[:,:] = 0.0
    basis = np.zeros(ndof, dtype=np.float64)
    jac = np.zeros((2,2), dtype=np.float64)
    jacinv = np.zeros((2,2), dtype=np.float64)

    for ig in range(quadrature.ng):
        # get quadrature points and weights
        x = quadrature.gp[ig,0]; y = quadrature.gp[ig,1]
        w = quadrature.gw[ig]

        # get basis function values and jacobian determinant
        elem.getBasisFunctions(x,y,basis)
        jdet = gmap.getJacobian(x,y,jac,jacinv)

        # physical location of quadrature point for coefficient function evaluation
        gx,gy = gmap.evalGeomMapping(x,y)

        # add contribution of this quadrature point to each integral
        for i in range(ndof):
            for j in range(ndof):
                localmass[i,j] += w * mass_coeff_func(gx,gy) * basis[i]*basis[j]*jdet

#@jit(nopython=True, cache=True)
def localL2Norm2(gmap, elem, quadrature, uvals):
    """ Computes the L2 norm squared of a FE function with dofs uvals on element elem.
	quadrature is the 2D quadrature contect to be used; has to be setup beforehand.
    """

    ndof = uvals.shape[0]
    localnorm2 = 0.0
    basis = np.zeros(ndof, dtype=np.float64)
    jac = np.zeros((2,2), dtype=np.float64)
    jacinv = np.zeros((2,2), dtype=np.float64)

    for ig in range(quadrature.ng):
        # get quadrature points and weights
        x = quadrature.gp[ig,0]; y = quadrature.gp[ig,1]
        w = quadrature.gw[ig]

        # get basis function values and jacobian determinant
        elem.getBasisFunctions(x,y,basis)
        jdet = gmap.getJacobian(x,y,jac,jacinv)

        # add contribution of this quadrature point to the integral
        dofsum = 0
        for i in range(ndof):
            dofsum += uvals[i] * basis[i]
        localnorm2 += w * dofsum*dofsum * jdet
    return localnorm2

def localL2Norm2_exactAtQuad(gmap, elem, quadrature, uvals):
    """ Computes the L2 norm squared of the error of FE solution with dofs uvals on element elem.
    Actual values of the exact solution function are used at the quadrature points.
	quadrature is the 2D quadrature contect to be used; has to be setup beforehand.
    """

    ndof = uvals.shape[0]
    localnorm2 = 0.0
    basis = np.zeros(ndof, dtype=np.float64)
    jac = np.zeros((2,2), dtype=np.float64)
    jacinv = np.zeros((2,2), dtype=np.float64)

    for ig in range(quadrature.ng):
        # get quadrature points and weights
        x = quadrature.gp[ig,0]; y = quadrature.gp[ig,1]
        w = quadrature.gw[ig]

        # get basis function values and jacobian determinant
        elem.getBasisFunctions(x,y,basis)
        jdet = gmap.getJacobian(x,y,jac,jacinv)

        # physical location of quadrature point for exact function evaluation
        gx,gy = gmap.evalGeomMapping(x,y)
        uexact = exact_sol(gx,gy)

        # add contribution of this quadrature point to the integral
        dofsum = 0
        for i in range(ndof):
            dofsum += uvals[i] * basis[i]
        localnorm2 += w * (dofsum-uexact)*(dofsum-uexact) * jdet
    return localnorm2

#@jit(nopython=True, cache=True)
def localLoadVector_domain(gmap, elem, quadrature, localload):
    """ Computes the domain integral part of the local load vector.
    localload must be pre-allocated.
    """

    ndof = localload.shape[0]
    localload[:] = 0.0
    basis = np.zeros(ndof, dtype=np.float64)
    jac = np.zeros((2,2), dtype=np.float64)
    jacinv = np.zeros((2,2), dtype=np.float64)

    for ig in range(quadrature.ng):
        # get quadrature points and weights
        x = quadrature.gp[ig,0]; y = quadrature.gp[ig,1]
        w = quadrature.gw[ig]

        # get basis gradients and jacobian determinant
        elem.getBasisFunctions(x,y,basis)
        jdet = gmap.getJacobian(x,y,jac,jacinv)

        # physical location of quadrature point for coefficient function evaluation
        gx,gy = gmap.evalGeomMapping(x,y)

        # add contribution of this quadrature point to each integral
        for i in range(ndof):
            localload[i] += w * rhs_func(gx,gy)*basis[i] * jdet

def localLoadVector_boundary(face, quadrature, localload):
    """ Computes the local boundary integral part of load vector for Neumann BCs.
    localload must be allocated before passing to this.
    g is a scalar function of two variables describing the Neumann BC.
    """
    pass

#@jit(nopython=True, cache=True)
def assemble(m, dirBCnum, A, b, pdeg, ngauss):
    """ Assembles a (dense, for now) LHS matrix and RHS vector.
        Applies a penalty method for Dirichlet BCs.
    """
    # For a Lagrange element, the number of DOFs per element is the same as the number of nodes per element

    elem = LagrangeTriangleElement()
    elem.setDegree(pdeg)

    gm = LagrangeTriangleMap()
    if(m.nnodel[0] == 6):
        gm.setDegree(2)
    elif(m.nnodel[0] == 3):
        gm.setDegree(1)

    integ2d = GLQuadrature2DTriangle(ngauss)

    print("assemble(): Beginning assembly loop over elements.")

    # iterate over the elements and add contributions
    for ielem in range(m.nelem):
        # setup required local arrays
        localmass = np.zeros((elem.ndof, elem.ndof))
        localstiff = np.zeros((elem.ndof, elem.ndof))
        localload = np.zeros(elem.ndof)
        phynodes = np.zeros((m.nnodel[ielem], 2))

        # set element
        phynodes[:,:] = m.coords[m.inpoel[ielem,:m.nnodel[ielem]],:]
        gm.setPhysicalElementNodes(phynodes)

        # get local matrices
        localLoadVector_domain(gm, elem, integ2d, localload)
        localStiffnessMatrix(gm, elem, integ2d, localstiff)
        localMassMatrix(gm, elem, integ2d, localmass)

        # add contributions to global
        b[m.inpoel[ielem,:m.nnodel[ielem]]] += localload[:]
        for i in range(m.nnodel[ielem]):
            #b[m.inpoel[ielem,i]] += localload[i]
            for j in range(m.nnodel[ielem]):
                A[m.inpoel[ielem,i], m.inpoel[ielem,j]] += localstiff[i,j] + localmass[i,j]


    # penalty for Dirichlet rows and columns
    """ For the row of each node corresponding to a Dirichlet boundary, multiply the diagonal entry by a huge number cbig,
        and set the RHS as boundary_value * cbig. This makes other entries in the row negligible, and the nodal value becomes
        (almost) equal to the required boundary value.
        I don't expect this to cause problems as the diagonal dominance of the matrix is increasing.
    """

    print("assembly(): Imposing penalties on Dirichlet rows")
    cbig = 1.0e30
    dirflags = np.zeros(m.npoin,dtype=np.int32)

    for iface in range(m.nbface):
        for inum in range(len(dirBCnum)):
            if m.bface[iface,m.nnofa[iface]] == dirBCnum[inum]:
                for inode in range(m.nnofa[iface]):
                    dirflags[m.bface[iface,inode]] = 1

    for ipoin in range(m.npoin):
        if dirflags[ipoin] == 1:
            A[ipoin,ipoin] *= cbig
            b[ipoin] = A[ipoin,ipoin]*dirichlet_function(m.coords[ipoin,0], m.coords[ipoin,1])


def removeDirichletRowsAndColumns(m,A,b,dirBCnum):
    """ Alternatively, rather than use a penalty method, we can eliminate Dirichlet rows and columns.
    """

    print("Removing Dirichlet rows and columns.")
    # preprocessing to detect Dirichlet nodes; dirflag stores whether a given node is a Dirichlet node
    ntotvars = m.npoin
    dirflag = np.zeros(m.npoin,dtype=np.int32)

    for iface in range(m.nbface):
        for inum in range(len(dirBCnum)):
            if(m.bface[iface,m.nnofa[iface]] == dirBCnum[inum]):

                # if this face is a Dirichlet face, mark its nodes
                for ibnode in range(m.nnofa[iface]):
                    dirflag[m.bface[iface,ibnode]] = 1

    for ipoin in range(m.npoin):
        ntotvars -= dirflag[ipoin]

    Ad = np.zeros((ntotvars,ntotvars), dtype=np.float64)
    bd = np.zeros(ntotvars, dtype=np.float64)

    inocc = 0
    for ipoin in range(m.npoin):
        if(dirflag[ipoin] != 1):
            bd[inocc] = b[ipoin]
            jnocc = 0
            for jpoin in range(m.npoin):
                if dirflag[jpoin] != 1:
                    Ad[inocc,jnocc] = A[ipoin,jpoin]
                    jnocc += 1
                else:
                    bd[inocc] -= ( A[ipoin,jpoin] * dirichlet_function(m.coords[jpoin,0],m.coords[jpoin,1]) )
            inocc += 1

    return (Ad,bd,dirflag)

def solveAndProcess(m, A, b, dirflag):

    print("solveAndProcess: Solving and getting final solution vector")
    xd = np.linalg.solve(A,b)
    x = np.zeros(m.npoin, dtype=np.float64)
    inocc = 0
    for ipoin in range(m.npoin):
        if(dirflag[ipoin] != 1):
            x[ipoin] = xd[inocc]
            inocc += 1
        else:
            x[ipoin] = dirichlet_function(m.coords[ipoin,0],m.coords[ipoin,1])
    print("solveAndProcess: Done.")
    return x,xd

#@jit(nopython=True, cache=True)
def compute_norm(m, v, pdeg, ngauss):
    """ Compute the L2 and H1 norms of the FE function v
    Note: it is currently assumed that all elements are topologically identical and use the same basis functions.
    """
    # For a Lagrange element, the number of DOFs per element is the same as the number of nodes per element

    elem = LagrangeTriangleElement()
    elem.setDegree(pdeg)

    gm = LagrangeTriangleMap()
    if(m.nnodel[0] == 6):
        gm.setDegree(2)
    elif(m.nnodel[0] == 3):
        gm.setDegree(1)

    integ2d = GLQuadrature2DTriangle(ngauss)

    l2norm = 0; h1norm = 0
    print("compute_norm(): Computing norms...")

    # iterate over the elements and add contributions
    for ielem in range(m.nelem):
        # setup required local arrays
        phynodes = np.zeros((m.nnodel[ielem], 2))

        # set element
        phynodes[:,:] = m.coords[m.inpoel[ielem,:m.nnodel[ielem]],:]
        gm.setPhysicalElementNodes(phynodes)
        uvals = v[m.inpoel[ielem,:m.nnodel[ielem]]]

        # compute and add contribution of this element
        l2normlocal = localL2Norm2(gm, elem, integ2d, uvals)
        l2norm += l2normlocal; h1norm += l2normlocal
        h1norm += localH1Seminorm2(gm, elem, integ2d, uvals)

    return (np.sqrt(l2norm), np.sqrt(h1norm))

def compute_norm_exact(m, v, pdeg, ngauss):
    """ Compute the L2 norm of the error of the FE solution v
    Note: it is currently assumed that all elements are topologically identical and use the same basis functions.
    """
    # For a Lagrange element, the number of DOFs per element is the same as the number of nodes per element

    elem = LagrangeTriangleElement()
    elem.setDegree(pdeg)

    gm = LagrangeTriangleMap()
    if(m.nnodel[0] == 6):
        gm.setDegree(2)
    elif(m.nnodel[0] == 3):
        gm.setDegree(1)

    integ2d = GLQuadrature2DTriangle(ngauss)

    l2norm = 0; h1norm = 0
    print("compute_norm(): Computing norms...")

    # iterate over the elements and add contributions
    for ielem in range(m.nelem):
        # setup required local arrays
        phynodes = np.zeros((m.nnodel[ielem], 2))

        # set element
        phynodes[:,:] = m.coords[m.inpoel[ielem,:m.nnodel[ielem]],:]
        gm.setPhysicalElementNodes(phynodes)
        uvals = v[m.inpoel[ielem,:m.nnodel[ielem]]]

        # compute and add contribution of this element
        l2normlocal = localL2Norm2_exactAtQuad(gm, elem, integ2d, uvals)
        l2norm += l2normlocal

    return np.sqrt(l2norm)
