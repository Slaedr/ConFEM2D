""" @brief Gauss-Lengendre quadrature rules for 1D and 2D integrals.
"""

import numpy as np
from numba import jitclass, int64, float64

spec = [('ng', int64), ('gp', float64[:,:]), ('gw', float64[:])]

class Quadrature:
    def __init__(self, ngauss):
        self.ng = ngauss
        self.gp = np.zeros((self.ng,2))
        self.gw = np.zeros(self.ng)

    def evaluate(self, fvals):
        """ Returns the integral. The number of entries in fvals must be ng."""
        #return (self.gw*fvals).sum()
        sum1 = 0.0
        for i in range(self.ng):
            sum1 += self.gw[i]*fvals[i]
        return sum1

#@jitclass(spec)
class GLQuadrature1D(Quadrature):
    def __init__(self, ngauss):
        self.ng = ngauss
        self.gp = np.zeros((self.ng,1), dtype=np.float64)
        self.gw = np.zeros(self.ng, dtype=np.float64)
        if self.ng == 1:
            self.gp[0,0] = 0.0
            self.gw[0] = 2.0
        elif self.ng == 2:
            self.gp[:,0] = [-1.0/np.sqrt(3), 1.0/np.sqrt(3)]
            self.gw[:] = [1.0, 1.0]
        elif self.ng == 3:
            self.gp[:,0] = [-np.sqrt(3.0/5.0), 0, np.sqrt(3.0/5.0)]
            self.gw[:] = [5.0/9.0, 8.0/9.0, 5.0/9.0]
        elif self.ng == 4:
            self.gp[:,0] = [-np.sqrt(3.0/7 + 2.0/7*np.sqrt(6.0/5)), -np.sqrt(3.0/7 - 2.0/7*np.sqrt(6.0/5)), np.sqrt(3.0/7 + 2.0/7*np.sqrt(6.0/5)), np.sqrt(3.0/7 + 2.0/7*np.sqrt(6.0/5))]
            self.gw[:] = [(18.0-np.sqrt(30))/36.0, (18.0+np.sqrt(30))/36.0, (18.0+np.sqrt(30))/36.0, (18.0-np.sqrt(30))/36.0]
        else:
            print("! GLQuadrature1D: Quadrature with this number of Gauss points is not supported!")

#@jitclass(spec)
class GLQuadrature2DTriangle(Quadrature):
    def __init__(self, ngauss):
        self.ng = ngauss
        self.gp = np.zeros((self.ng,2), dtype=np.float64)
        self.gw = np.zeros(self.ng, dtype=np.float64)
        if self.ng == 1:
            self.gp[0,:] = [1.0/3, 1.0/3]
            self.gw[0] = 0.5
            #print("GLQuadrature2DTriangle: Ngauss = 1.")
        elif self.ng == 3:
            self.gp[:,:] = np.array([0.6666666666667,0.1666666666667 , 0.1666666666667,0.6666666666667,  0.1666666666667,0.1666666666667]).reshape((self.ng,2))
            self.gw[:] = [0.1666666666667, 0.1666666666667, 0.1666666666667]
            #print("GLQuadrature2DTriangle: Ngauss = 3.")
        elif self.ng == 4:
            self.gp = np.array([0.33333333333,0.33333333333,  0.20000000000,0.20000000000,  0.20000000000, 0.60000000000,  0.60000000000, 0.20000000000]).reshape((self.ng,2))
            self.gw[:] = [-0.28125000000, 0.26041666667, 0.26041666667, 0.26041666667]
            #print("GLQuadrature2DTriangle: Ngauss = 4.")
        elif self.ng == 6:
            self.gp[:,:] = np.array([0.108103018168070,0.445948490915965,
                    0.445948490915965,0.108103018168070,
                    0.445948490915965,0.445948490915965,
                    0.816847572980459,0.091576213509771,
                    0.091576213509771,0.816847572980459,
                    0.091576213509771,0.091576213509771]).reshape((self.ng,2))
            self.gw[:] = [0.1116907948390055,
                       0.1116907948390055,
                       0.1116907948390055,
                       0.0549758718276610,
                       0.0549758718276610,
                       0.0549758718276610]
            #print("GLQuadrature2DTriangle: Ngauss = 6.")
        elif self.ng == 12:
            self.gp[:,:] = np.array([0.873821971016996,0.063089014491502,
                      0.063089014491502,0.873821971016996,
                      0.063089014491502,0.063089014491502,
                      0.501426509658179,0.249286745170910,
                      0.249286745170910,0.501426509658179,
                      0.249286745170910,0.249286745170910,
                      0.636502499121399,0.310352451033785,
                      0.636502499121399,0.053145049844816,
                      0.310352451033785,0.636502499121399,
                      0.310352451033785,0.053145049844816,
                      0.053145049844816,0.310352451033785,
                      0.053145049844816,0.636502499121399]).reshape((self.ng,2))
            self.gw[:] = [0.0254224531851035,
                       0.0254224531851035,
                       0.0254224531851035,
                       0.0583931378631895,
                       0.0583931378631895,
                       0.0583931378631895,
                       0.0414255378091870,
                       0.0414255378091870,
                       0.0414255378091870,
                       0.0414255378091870,
                       0.0414255378091870,
                       0.0414255378091870]
            #print("GLQuadrature2DTriangle: Ngauss = 12.")
        else:
            print("! GLQuadrature2DTriangle: Quadrature with this number of Gauss points is not supported!")
