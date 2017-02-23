""" @brief Setup for finite elements
"""

import numpy as np
from numba import jit, generated_jit, jitclass, int32, float64
from mesh import *
from quadrature import GLQuadrature1D, GLQuadrature2DTriangle

spec = [('nnodel', int32), ('refnodes', float64[:,:]), ('phynodes', float64[:,:])]

class Element:
    """ @brief Abstract class for a finite element.
    Members are
    nnodel: number of nodes in the element
    phynodes: locations of physical nodes
    refnodes: locations of reference nodes
    """
    def __init__(self):
        pass
    def setPhysicalElementNodes(self, physical_nodes):
        """ physical_nodes is a nnodel x ndim numpy array describing locations of the physical nodes."""
        self.phynodes = np.copy(physical_nodes)
        #self.phynodes = physical_nodes[:,:]
    def setReferenceElementNodes(self):
        pass
    def evalGeomMapping(self,x,y):
        """ Returns the physical coordinates of reference location (x,y) in the reference element.
        """
        pass
    def getJacobian(self, x, y, jac, jacinv):
        """ The ndim x ndim array jac contains the Jacobian matrix of the geometric mapping on return
        and jacinv contains its inverse.
        Returns the value of the determinant.
        """
        pass
    def getBasisFunctions(self, x, y, bvals):
        """ Returns the basis function value on the reference element as a function of reference coordinates.
        bvals must be preallocated as ndofs x 1. On return, it contains values of the basis function at (x,y).
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

    def evalGeomMapping(self,x,y):
        rg = np.zeros(2)
        rg[:] = self.phynodes[0,:]*(1.0-x-y) + self.phynodes[1,:]*x + self.phynodes[2,:]*y
        return (rg[0],rg[1])

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

    def evalGeomMapping(self,x,y):
        rg = np.zeros(2)
        rg[:] = self.phynodes[0,:]*(1.0-3*x-3*y+2*x*x+2*y*y+4*x*y) + self.phynodes[1,:]*(2.0*x*x-x) + self.phynodes[2,:]*(2.0*y*y-y) + \
                self.phynodes[3,:]*4.0*(x-x*x-x*y) + self.phynodes[4,:]*4.0*x*y + self.phynodes[5,:]*4.0*(y-y*y-x*y)
        return (rg[0],rg[1])

    def getJacobian(self, x, y, jac, jacinv):
        jac[:,0] = self.phynodes[0,:]*(-3+4*x+4*y) +self.phynodes[1,:]*(4*x-1) +self.phynodes[3,:]*4*(1-2*x-y) +self.phynodes[4,:]*4*y -self.phynodes[5,:]*4.0*y
        jac[:,1] = self.phynodes[0,:]*(-3+4*y+4*x) +self.phynodes[2,:]*(4*y-1) -self.phynodes[3,:]*4*x +self.phynodes[4,:]*4*x +self.phynodes[5,:]*4*(1-2*y-x)
        jdet = jac[0,0]*jac[1,1] - jac[0,1]*jac[1,0]
        jacinv[0,0] = jac[1,1]/jdet; jacinv[0,1] = -jac[0,1]/jdet
        jacinv[1,0] = -jac[1,0]/jdet; jacinv[1,1] = jac[0,0]/jdet
        return jdet

#@jitclass(spec)
class LagrangeP1TriangleElement(P1TriangleElement):
    """ Triangular element with Lagrange P1 basis for the trial/test space.
    """
    def __init__(self):
        print("Initialized Lagrange P1 triangle element.")

    def getBasisFunctions(self, x, y, bvals):
        bvals[:] = [1.0-x-y, x, y]

    def getBasisGradients(self, x, y, bgrvals):
        bgrvals[0,:] = [-1.0, -1.0]
        bgrvals[1,:] = [1.0, 0.0]
        bgrvals[2,:] = [0.0, 1.0]

#@jitclass(spec)
class LagrangeP2TriangleElement(P2TriangleElement):
    """ Triangular element with Lagrange P2 basis for the trial/test space.
    NOTE: The 2 functions below need to be checked for curved elements - ie,
    what is B(T(x)), when T is a non-affine geometric map and x is the reference coordinate?
    """
    def __init__(self):
        print("Initialized Lagrange P2 triangle element.")

    def getBasisFunctions(self, x, y, bvals):
        bvals[0] = 1.0 - 3*x - 3*y + 2*x*x + 4*x*y + 2*y*y
        bvals[1] = 2.0*x*x - x
        bvals[2] = 2.0*y*y - y
        bvals[3] = 4.0*(x - x*x - x*y)
        bvals[4] = 4.0*x*y
        bvals[5] = 4.0*(y - y*y - x*y)

    def getBasisGradients(self, x, y, bgrvals):
        bgrvals[0,:] = [-3.0+4.0*x+4.0*y, -3.0+4.0*x+4.0*y]
        bgrvals[1,:] = [4.0*x-1.0, 0.0]
        bgrvals[2,:] = [0.0, 4.0*y-1.0]
        bgrvals[3,:] = [4.0*(1-2.0*x-y), -4.0*x]
        bgrvals[4,:] = [4.0*y, 4.0*x]
        bgrvals[5,:] = [-4.0*y, 4.0*(1.0-2.0*y-x)]


if __name__ == "__main__":
    elem = LagrangeP2TriangleElement()
