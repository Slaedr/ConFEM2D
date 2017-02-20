""" @brief Assembly routines of local and global FE matrices.
"""

import numpy as np
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
    pass

@jit(nopython=True)
def mass_coeff_func(x,y):
    pass

@jit(nopython=True)
def exact_sol(x,y):
    pass

@jit(nopython=True)
def dirichlet_value(x,y):
    pass

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

        # add contribution of this quadrature point to each integral
        for i in range(ndof):
            for j in range(ndof):
                localmass[i,j] += w * np.dot( np.dot(jacinv.T, basisg[i,:]), np.dot(jacinv.T, basisg[j,:]) ) * jdet

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

        # add contribution of this quadrature point to each integral
        for i in range(ndof):
            for j in range(ndof):
                localmass[i,j] += w * basis[i]*basis[j]*jdet

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

        # add contribution of this quadrature point to each integral
        for i in range(ndof):
            localload[i] += w * rhs_func(x,y)*basis[i] * jdet

def localLoadVector_boundary(face, quadrature, localload):
    """ Computes the local boundary integral part of load vector for Neumann BCs.
    localload must be allocated before passing to this.
    g is a scalar function of two variables describing the Neumann BC.
    """
    pass

#@jit(nopython=True, cache=True)
def assemble(m, dirBCnum):
    # For a Lagrange element, the number of DOFs per element is the same as the number of nodes per element

    A = np.zeros((m.npoin, m.npoin), dtype=np.float64)
    b = np.zeros(m.npoin, dtype=np.float64)

    if(m.nnodel == 6):
        elem = LagrangeP2TriangleElement()
        ngauss = 6
    elif(m.nnodel == 3):
        elem = LagrangeP1TriangleElement()
        ngauss = 3
    integ2d = Quadrature2DTriangle(ngauss)

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

    # remove Dirichlet rows and columns
    
    Ad = np.zeros((ntotvars,ntotvars), dtype=np.float64)
    bd = np.zeros(ntotvars, dtype=np.float64)

    inocc = 0
    for ipoin in range(m.npoin):
        if(dirflag[ipoin] != 1):
            jnocc = 0
            for jpoin in range(m.npoin):
                if dirflag[jpoin] != 1:
                    Ad[inocc,jnocc] = A[ipoin,jpoin]
                    jnocc += 1
                else:
                    bd[inocc] -= ( A[ipoin,jpoin] * dirichlet_function(m.coords[jpoin,0],m.coords[jpoin,1]) )
            inocc += 1

    return (Ad,bd,dirflag)


if __name__ == "__main__":
    m = Mesh2d()
    m.readGmsh("try.msh")
    dirBCflag = 2
