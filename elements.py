""" @brief Setup for finite elements
"""

import numpy as np
from numba import jit, generated_jit, jitclass, int32, float64
from mesh import *
from quadrature import GLQuadrature1D, GLQuadrature2DTriangle

spec = [('nnodel', int32), ('phynodes', float64[:,:])]

class GeometricMap:
    """ @brief Abstract class for mapping between a physical element and a reference element.
    """
    def setDegree(self, deg):
        self.degree = deg

    def setPhysicalElementNodes(self, physical_nodes):
        """ physical_nodes is a nnodel x ndim numpy array describing locations of the physical nodes."""
        self.phynodes = np.copy(physical_nodes)

    def evalGeomMapping(self,x,y):
        """ Returns the physical coordinates of reference location (x,y) in the reference element.
        """
        pass

    def getJacobian(self, x, y, jac, jacinv):
        """ The ndim x ndim array jac contains the Jacobian matrix of the geometric mapping on return
        and jacinv contains its inverse, evaluated at the reference coordinates (x,y).
        Returns the value of the determinant.
        """
        pass
    def getJacobianDeterminant(self, x, y):
        """ Returns the value of the determinant.
        """
        pass

class P1Triangle(GeometricMap):

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

class P2Triangle(GeometricMap):

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

class LagrangeTriangleMap(GeometricMap):
    """ @brief Mapping from reference to physical element based on Lagrange basis functions.
    """
    def evalGeomMapping(self,x,y):
        rg = np.zeros(2)
        if self.degree == 1:
            rg[:] = self.phynodes[0,:]*(1.0-x-y) + self.phynodes[1,:]*x + self.phynodes[2,:]*y
        elif self.degree == 2:
            rg[:] = self.phynodes[0,:]*(1.0-3*x-3*y+2*x*x+2*y*y+4*x*y) + self.phynodes[1,:]*(2.0*x*x-x) + self.phynodes[2,:]*(2.0*y*y-y) + \
                    self.phynodes[3,:]*4.0*(x-x*x-x*y) + self.phynodes[4,:]*4.0*x*y + self.phynodes[5,:]*4.0*(y-y*y-x*y)
        return (rg[0],rg[1])

    def getJacobian(self, x, y, jac, jacinv):
        if self.degree == 1:
            jac[:,0] = self.phynodes[1,:]-self.phynodes[0,:]
            jac[:,1] = self.phynodes[2,:]-self.phynodes[0,:]
            jdet = jac[0,0]*jac[1,1] - jac[0,1]*jac[1,0]
            jacinv[0,0] = jac[1,1]/jdet; jacinv[0,1] = -jac[0,1]/jdet
            jacinv[1,0] = -jac[1,0]/jdet; jacinv[1,1] = jac[0,0]/jdet
        elif self.degree == 2:
            jac[:,0] = self.phynodes[0,:]*(-3+4*x+4*y) +self.phynodes[1,:]*(4*x-1) +self.phynodes[3,:]*4*(1-2*x-y) +self.phynodes[4,:]*4*y -self.phynodes[5,:]*4.0*y
            jac[:,1] = self.phynodes[0,:]*(-3+4*y+4*x) +self.phynodes[2,:]*(4*y-1) -self.phynodes[3,:]*4*x +self.phynodes[4,:]*4*x +self.phynodes[5,:]*4*(1-2*y-x)
            jdet = jac[0,0]*jac[1,1] - jac[0,1]*jac[1,0]
            jacinv[0,0] = jac[1,1]/jdet; jacinv[0,1] = -jac[0,1]/jdet
            jacinv[1,0] = -jac[1,0]/jdet; jacinv[1,1] = jac[0,0]/jdet
        return jdet

    def getJacobianDeterminant(self):
        jac = np.zeros((2,2),dtype=np.float64)
        jdet = 0
        if self.degree == 1:
            jac[:,0] = self.phynodes[1,:]-self.phynodes[0,:]
            jac[:,1] = self.phynodes[2,:]-self.phynodes[0,:]
            jdet = jac[0,0]*jac[1,1] - jac[0,1]*jac[1,0]
        elif self.degree == 2:
            jac[:,0] = self.phynodes[0,:]*(-3+4*x+4*y) +self.phynodes[1,:]*(4*x-1) +self.phynodes[3,:]*4*(1-2*x-y) +self.phynodes[4,:]*4*y -self.phynodes[5,:]*4.0*y
            jac[:,1] = self.phynodes[0,:]*(-3+4*y+4*x) +self.phynodes[2,:]*(4*y-1) -self.phynodes[3,:]*4*x +self.phynodes[4,:]*4*x +self.phynodes[5,:]*4*(1-2*y-x)
            jdet = jac[0,0]*jac[1,1] - jac[0,1]*jac[1,0]
        return jdet


class Element:
    """ @brief Abstract class for a finite element with basis functions defined on the reference element.
    Members are
    nnodel: number of nodes in the element
    phynodes: locations of physical nodes
    """

    def setDegree(self, deg):
        """ Set the polynomial degree of trial and test basis functions.
        Must be overriden by child classes for ndof computation."""
        self.degree = deg
        self.ndof = 1

    def getBasisFunctions(self, x, y, bvals):
        """ Returns the basis function value on the reference element as a function of reference coordinates.
        bvals must be preallocated as ndofs x 1. On return, it contains values of the basis function at (x,y).
        """
        pass

    def getBasisGradients(self, x, y, bgrvals):
        """ bgrvals must be ndofs x ndim. Contains partial derivatives of the basis functions on return.
        """
        pass

#@jitclass(spec)
class LagrangeTriangleElement(Element):
    """ Triangular element with Lagrange P1 basis for the trial/test space.
    """
    def __init__(self):
        print("Initialized Lagrange triangle element.")

    def setDegree(self, deg):
        self.degree = deg
        self.ndof = 1
        if self.degree == 1:
            self.ndof = 3
        elif self.degree == 2:
            self.ndof = 6

    def getBasisFunctions(self, x, y, bvals):
        if self.degree == 1:
            bvals[:] = [1.0-x-y, x, y]
        elif self.degree == 2:
            bvals[0] = 1.0 - 3*x - 3*y + 2*x*x + 4*x*y + 2*y*y
            bvals[1] = 2.0*x*x - x
            bvals[2] = 2.0*y*y - y
            bvals[3] = 4.0*(x - x*x - x*y)
            bvals[4] = 4.0*x*y
            bvals[5] = 4.0*(y - y*y - x*y)

    def getBasisGradients(self, x, y, bgrvals):
        if self.degree == 1:
            bgrvals[0,:] = [-1.0, -1.0]
            bgrvals[1,:] = [1.0, 0.0]
            bgrvals[2,:] = [0.0, 1.0]
        elif self.degree == 2:
            bgrvals[0,:] = [-3.0+4.0*x+4.0*y, -3.0+4.0*x+4.0*y]
            bgrvals[1,:] = [4.0*x-1.0, 0.0]
            bgrvals[2,:] = [0.0, 4.0*y-1.0]
            bgrvals[3,:] = [4.0*(1-2.0*x-y), -4.0*x]
            bgrvals[4,:] = [4.0*y, 4.0*x]
            bgrvals[5,:] = [-4.0*y, 4.0*(1.0-2.0*y-x)]


if __name__ == "__main__":
    gm = LagrangeMap()
    gm.setDegree(1)
    pn = np.array([[0,1],[2,0],[0,2.0]])
    gm.setPhysicalElementNodes(pn)
    jac = np.zeros((2,2))
    jacinv = np.zeros((2,2))
    jacdet = gm.getJacobian(0,0.6,jac,jacinv)
    print(jac)

    elem = LagrangeTriangleElement()
    elem.setDegree(2)
    bvals = np.zeros(6)
    elem.getBasisFunctions(0,0.6, bvals)
    print(bvals)
