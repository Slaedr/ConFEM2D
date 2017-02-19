""" @brief Assembly routines of local and global FE matrices.
"""

import numpy as np
import numpy.linalg
from numba import jit, generated_jit
from mesh import *
from quadrature import GLQuadrature1D, GLQuadrature2DTriangle

class Element:
    """ @brief Abstract class for a finite element.
    Members are
    nnodel: number of nodes in the element
    phynodes: locations of physical nodes
    refnodes: locations of reference nodes
    """
    def __init__(self):
        pass
    def setPhysicalElementNodes(self, physical_nodes, num_dofs):
        """ physical_nodes is a nnodel x ndim numpy array describing locations of the physical nodes."""
        self.phynodes = np.copy(pysical_nodes)
        self.ndofs = num_dofs
    def setReferenceElementNodes(self):
        pass
    def getJacobianDet(self, x, y):
        pass
    def getJacobian(self, x, y, jac, jacinv):
        """ The ndim x ndim array jac contains the Jacobian matrix of the geometric mapping on return
        and jacinv contains its inverse.
        Returns the value of the determinant.
        """
        pass
    def getBasisFunctions(self, x, y, bvals):
        """ bvals must be preallocated as ndofs x 1. On return, it contains values of the basis function at (x,y).
        """
        pass
    def getBasisGradients(self, x, y, bgrvals):
        """ bgrvals must be ndofs x ndim. Contains partial derivatives of the basis functions on return.
        """
        pass

class P1TriangleElement(Element):
    def setReferenceElementNodes(self):
        self.refnodes = np.zeros(phy_nodes)
        self.nnodel = phy_nodes.shape[0]
        if self.nnodel == 3:
            refnodes[0,:] = [0.0,0.0]
            refnodes[1,:] = [1.0,0.0]
            refnodes[2,:] = [0.0,1.0]
        else:
            print("! P1TriangleElement: Element with " + str(self.nnodel) + " nodes not available!")

    def getJacobian(self, x, y, jac, jacinv):
        jac[:,0] = self.phynodes[1,:]-self.phynodes[0,:]
        jac[:,1] = self.phynodes[2,:]-self.phynodes[0,:]
        jdet = jac[0,0]*jac[1,1] - jac[0,1]*jac[1,0]
        jacinv[0,0] = jac[1,1]/jdet; jacinv[0,1] = -jac[0,1]/jdet
        jacinv[1,0] = -jac[1,0]/jdet; jacinv[1,1] = jac[0,0]/jdet
        return jdet

class P2TriangleElement(Element):
    def setReferenceElementNodes(self):
        self.refnodes = np.zeros(phy_nodes)
        self.nnodel = phy_nodes.shape[0]
        if self.nnodel == 6:
            refnodes[3,:] = (refnodes[0,:]+refnodes[1,:])*0.5
            refnodes[4,:] = (refnodes[1,:]+refnodes[2,:])*0.5
            refnodes[5,:] = (refnodes[2,:]+refnodes[0,:])*0.5
        else:
            print("! P2TriangleElement: Element with " + str(self.nnodel) + " nodes not available!")

    def getJacobian(self, x, y, jac, jacinv):
        jac[:,0] = self.phynodes[0,:]*(-3-4*x-4*y) +self.phynodes[1,:]*(4*x-1) +self.phynodes[3,:]*4*(1-2*x-y) +self.phynodes[4,:]*4*y -self.phynodes[5,:]*y
        jac[:,1] = self.phynodes[0,:]*(-3-4*y-4*x) +self.phynodes[2,:]*(4*y-1) -self.phynodes[3,:]*4*x +self.phynodes[4,:]*4*x +self.phynodes[5,:]*4*(1-2*y-x)
        jdet = jac[0,0]*jac[1,1] - jac[0,1]*jac[1,0]
        jacinv[0,0] = jac[1,1]/jdet; jacinv[0,1] = -jac[0,1]/jdet
        jacinv[1,0] = -jac[1,0]/jdet; jacinv[1,1] = jac[0,0]/jdet
        return jdet

class LagrangeP1TriangleElement(P1TriangleElement):
    """ Triangular element with Lagrange P1 basis for the trial/test space.
    """
    def getBasisFunctions(self, x, y, bvals):
        bvals[:] = [1.0-x-y, x, y]

    def getBasisGradients(self, x, y, bgrvals):
        bgrvals[0,:] = [-1.0, -1.0]
        bgrvals[1,:] = [1.0, 0.0]
        bgrvals[2,:] = [0.0, 1.0]

class LagrangeP2TriangleElement(P2TriangleElement):
    """ Triangular element with Lagrange P2 basis for the trial/test space.
    NOTE: The 2 functions below need to be checked - ie,
    what is B(T(x)), where T is the geometric map and x is the reference coordinate?
    """
    def getBasisFunctions(self, x, y, bvals):
        bvals[0] = 1.0 - 3*x - 3*y - 2*x*x - 2*y*y - 4*x*y
        bvals[1] = 2.0*x*x - x
        bvals[2] = 2.0*y*y - y
        bvals[3] = 4.0*(x - x*x - x*y)
        bvals[4] = 4.0*x*y
        bvals[5] = 4.0*(y - y*y - x*y)

    def getBasisGradients(self, x, y, bgrvals):
        bgrvals[0,:] = [-3.0-4.0*x-4.0*y, -3.0-4.0*x-4.0*y]
        bgrvals[1,:] = [4.0*x-1.0, 0.0]
        bgrvals[2,:] = [0.0, 4.0*y-1.0]
        bgrvals[3,:] = [4.0*(1-2.0*x-y), -4.0*x]
        bgrvals[4,:] = [4.0*y, 4.0*x]
        bgrvals[5,:] = [-4.0*y, 4.0*(1.0-2.0*y-x)]


#@jit(nopython=True, cache=True)
def coeff_mass(x, y, elem):
    return x + 2.0*y

#@jit(nopython=True, cache=True)
def coeff_stiffness(x, y):
    return x*x + y*y - x*y

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

def localLoadVector_domain(elem, quadrature, localload):
    """ Computes the domain integral part of the local load vector.
    localload must be pre-allocated.
    """
    pass

def localLoadVector_boundary(face, quadrature, localload):
    """ Computes the local boundary integral part of load vector for Neumann BCs.
    localload must be allocated before passing to this.
    """
    pass
