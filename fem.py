""" @brief Assembly routines of local and global FE matrices.
"""

import numpy as np
from numpy.linalg import solve
from numba import jit, generated_jit, jitclass, int32, float64
from mesh import *
from quadrature import GLQuadrature1D, GLQuadrature2DTriangle
from elements import *

# function prescribing the known RHS f
@jit(nopython=True)
def rhs_func(x,y):
    pass

@jit(nopython=True)
def stiffness_coeff_func(x,y):
    return 1.0

@jit(nopython=True)
def mass_coeff_func(x,y):
    return 1.0

@jit(nopython=True)
def exact_sol(x,y):
    pass

@jit(nopython=True)
def dirichlet_value(x,y):
    return 0.0

#@jit(nopython=True, cache=True)
def localStiffnessMatrix(elem, quadrature, localstiff):
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
        jdet = elem.getJacobian(x,y,jac,jacinv)

        # physical location of quadrature point for coefficient function evaluation
        gx,gy = elem.evalGeomMapping(x,y)

        # add contribution of this quadrature point to each integral
        for i in range(ndof):
            for j in range(ndof):
                localmass[i,j] += w * stiffness_coeff_func(gx,gy) * np.dot( np.matmul(jacinv.T, basisg[i,:]), np.matmul(jacinv.T, basisg[j,:]) ) * jdet

#@jit(nopython=True, cache=True)
def localH1Seminorm2(elem, quadrature, uvals):
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
        jdet = elem.getJacobian(x,y,jac,jacinv)

        # physical location of quadrature point for coefficient function evaluation
        gx,gy = elem.evalGeomMapping(x,y)

        # add contribution of this quadrature point to the integral
        dofsum1 = np.array([0.0,0.0]); dofsum2 = np.array([0.0, 0.0])
        for i in range(ndof):
            dofsum1[:] += uvals[i] * np.matmul(jacinv.T, basisg[i,:])
            dofsum2[:] += uvals[i] * np.matmul(jacinv.T, basisg[i,:])
        localseminorm2 += np.dot(dofsum1,dofsum2) * w * jdet

    return localseminorm2

#@jit(nopython=True, cache=True)
def localMassMatrix(elem, quadrature, localmass):
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
        jdet = elem.getJacobian(x,y,jac,jacinv)
        
        # physical location of quadrature point for coefficient function evaluation
        gx,gy = elem.evalGeomMapping(x,y)

        # add contribution of this quadrature point to each integral
        for i in range(ndof):
            for j in range(ndof):
                localmass[i,j] += w * mass_coeff_func(gx,gy) * basis[i]*basis[j]*jdet

#@jit(nopython=True, cache=True)
def localL2Norm2(elem, quadrature, uvals):
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
        jdet = elem.getJacobian(x,y,jac,jacinv)
        
        # physical location of quadrature point for coefficient function evaluation
        gx,gy = elem.evalGeomMapping(x,y)

        # add contribution of this quadrature point to the integral
        dofsum = 0
        for i in range(ndof):
            dofsum += uvals[i] * basis[i]
        localnorm2 += w * dofsum*dofsum * jdet
    return localnorm2

#@jit(nopython=True, cache=True)
def localLoadVector_domain(elem, quadrature, localload):
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
        jdet = elem.getJacobian(x,y,jac,jacinv)

        # physical location of quadrature point for coefficient function evaluation
        gx,gy = elem.evalGeomMapping(x,y)

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
def assemble(m, dirBCnum, A, b):
    """ Assembles a (dense, for now) LHS matrix and RHS vector.
        Applies a penalty method for Dirichlet BCs.
    """
    # For a Lagrange element, the number of DOFs per element is the same as the number of nodes per element

    if(m.nnodel == 6):
        elem = LagrangeP2TriangleElement()
        ngauss = 6
    elif(m.nnodel == 3):
        elem = LagrangeP1TriangleElement()
        ngauss = 3
    integ2d = Quadrature2DTriangle(ngauss)

    # iterate over the elements and add contributions
    for ielem in range(m.nelem):
        # setup required local arrays
        localmass = np.zeros((m.nnodel[ielem],m.nnodel[ielem]))
        localstiff = np.zeros((m.nnodel[ielem], m.nnodel[ielem]))
        localload = np.zeros(m.nnodel[ielem])
        phynodes = np.zeros((m.nnodel[ielem], 2))

        # set element
        phynodes[:,:] = m.coords[m.inpoel[ielem,:],:]
        elem.setPhysicalElementNodes(phynodes)

        # get local matrices
        localLoadVector_domain(elem, integ2d, localload)
        localStiffnessMatrix(elem, integ2d, localstiff)
        localMassMatrix(elem, integ2d, localmass)

        # add contributions to global
        A[m.inpoel[ielem,:], m.inpoel[ielem,:]] += localstiff[:,:] + localmass[:,:]
        b[m.inpoel[ielem,:]] += localload[:]

    # penalty for Dirichlet rows and columns
    """ For the row of each node corresponding to a Dirichlet boundary, multiply the diagonal entry by a huge number cbig,
        and set the RHS as boundary_value * cbig. This makes other entries in the row negligible, and the nodal value becomes
        (almost) equal to the required boundary value.
        I don't expect this to cause problems as the diagonal dominance of the matrix is increasing.
    """

    cbig = 1.0e30
    for iface in range(m.nbface):
        if m.bface[iface,m.nnofa[iface]] == dirBCnum:
            for inode in range(m.nnofa[iface]):
                node = m.bface[iface,inode]
                A[node,node] *= cbig
                b[node] = cbig*dirichlet_function(m.coords[node,0], m.coords[node,1])

    return (A,b)
    
def removeDirichletRowsAndColumns(m,A,b):
    """ Alternatively, rather than use a penalty method, we can eliminate Dirichlet rows and columns.
    """
    
    # preprocessing to detect Dirichlet nodes; dirflag stores whether a given node is a Dirichlet node
    ntotvars = m.npoin
    dirflag = np.zeros(m.npoin,dtype=np.int32)

    for iface in range(m.nbface):
        if(m.bface[iface,m.nnofa[iface]] == dirBCnum):
            
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

def solveAndProcess(m, A, b):

    x = solve(A,b)

#@jit(nopython=True, cache=True)
def compute_norm(m, v):
    """ Compute the L2 and H1 norms of the FE function v
    Note: it is currently assumed that all elements are topologically identical and use the same basis functions.
    """
    # For a Lagrange element, the number of DOFs per element is the same as the number of nodes per element

    if(m.nnodel == 6):
        elem = LagrangeP2TriangleElement()
        ngauss = 6
    elif(m.nnodel == 3):
        elem = LagrangeP1TriangleElement()
        ngauss = 3
    integ2d = Quadrature2DTriangle(ngauss)

    l2norm = 0; h1norm = 0

    # iterate over the elements and add contributions
    for ielem in range(m.nelem):
        # setup required local arrays
        phynodes = np.zeros((m.nnodel[ielem], 2))

        # set element
        phynodes[:,:] = m.coords[m.inpoel[ielem,:],:]
        elem.setPhysicalElementNodes(phynodes)
        uvals = v[m.inpoel[ielem,:]]

        # compute and add contribution of this element
        l2normlocal = localL2Norm2(elem, integ2d, uvals)
        l2norm += l2normlocal; h1norm += l2normlocal
        h1norm += localH1Seminorm2(elem, integ2d, uvals)

    return (np.sqrt(l2norm), np.sqrt(h1norm))

