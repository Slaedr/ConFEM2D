""" @brief 2D Mesh handling class
"""

import gc
import numpy as np
from numba import jit

class Mesh2d:
    """ @brief Handles the mesh data.

    Contains:
    - npoin: number of vertices in the mesh
    - nelem: number of elements
    - nbface: number of boundary faces
    - maxnnodel: max number of nodes per element
    - nbtags: number of boundary tags for each boundary face
    - ndtags number of domain tags for each element
    - nnodel: array containing number of nodes in each element
    - nnofa: array containing number of nodes in each face
    - coords: array containing coordinates of mesh vertices
    - inpoel: interconnecitivity matrix; node numbers (in coords) of the nodes making up each element
    - dtags: array containing domain marker tags
    - bface: array containing vertex numbers of vertices in each boundary face, as well as two boundary markers per boundary face
    - dtag: array containing domain markers for each element
    """
    nbtags = 2
    ndtags = 2
    def __init__(self):
        self.npoin = 0
        self.nelem = 0
        self.maxnnodel = 4

    def readGmsh(self, fname):
        """ Reads a Gmsh2 mesh file."""
        f = open(fname,'r')
        for i in range(4):
            f.readline()
        
        self.npoin = int(f.readline())
        print("Mesh2d: readGmsh(): Num points = " + str(self.npoin))
        temp = np.fromfile(f, dtype=float, count=self.npoin*4, sep=" ").reshape((self.npoin,4))
        self.coords = temp[:,1:-1]
        print("Mesh2d: readGmsh(): Coords read.")

        for i in range(2):
            f.readline()
        nallelem = int(f.readline())
        allelems = np.zeros((nallelem,Mesh2d.ndtags+12))
        self.nbface = 0
        self.nelem = 0

        # first we just read everything in the order given
        for i in range(nallelem):
            # store everything but the first entry in the line
            elem = np.array(f.readline().split(),dtype=int)[1:]
            if elem[0] == 1 or elem[0] == 8:
                self.nbface += 1
            elif elem[0] == 2 or elem[0]==9 or elem[0] == 3 or elem[0]==10:
                self.nelem += 1
            else:
                print("Mesh2d: readGmsh(): ! Invalid element type!")
            for j in range(len(elem)):
                allelems[i,j] = elem[j]

        self.bface = np.zeros((self.nbface,2+Mesh2d.nbtags),dtype=int)
        self.inpoel = np.zeros((self.nelem, self.maxnnodel), dtype=int)
        self.dtags = np.zeros((self.nelem,self.ndtags), dtype=int)
        self.nnodel = np.zeros(self.nelem,dtype=int)
        self.nnofa = np.zeros(self.nbface,dtype=int)
        iface = 0; ielem = 0

        for i in range(nallelem):
            if allelems[i,0] == 1:
                # P1 line segment
                self.nnofa[iface] = 2
                self.bface[iface, :self.nnofa[iface]] = allelems[i, 2+Mesh2d.nbtags:2+Mesh2d.nbtags+self.nnofa[iface]]-1
                self.bface[iface, self.nnofa[iface]:self.nnofa[iface]+Mesh2d.nbtags] = allelems[i, 2:2+Mesh2d.nbtags]
                iface += 1
            elif allelems[i,0] == 8:
                # P2 line segment
                self.nnofa[iface] = 3
                self.bface[iface, :self.nnofa[iface]] = allelems[i, 2+Mesh2d.nbtags:2+Mesh2d.nbtags+self.nnofa[iface]]-1
                self.bface[iface, self.nnofa[iface]:self.nnofa[iface]+Mesh2d.nbtags] = allelems[i, 2:2+Mesh2d.nbtags]
                iface += 1
            elif allelems[i,0] == 2:
                # P1 tri
                self.nnodel[ielem] = 3
                self.inpoel[ielem, :self.nnodel[ielem]] = allelems[i, 2+Mesh2d.ndtags:2+Mesh2d.ndtags+self.nnodel[ielem]]-1
                self.dtags[ielem, :Mesh2d.ndtags] = allelems[i, 2:2+Mesh2d.ndtags]
                ielem += 1
            elif allelems[i,0] == 9:
                # P2 tri
                self.nnodel[ielem] = 6
                self.inpoel[ielem, :self.nnodel[ielem]] = allelems[i, 2+Mesh2d.ndtags:2+Mesh2d.ndtags+self.nnodel[ielem]]-1
                self.dtags[ielem, :Mesh2d.ndtags] = allelems[i, 2:2+Mesh2d.ndtags]
                ielem += 1
            elif allelems[i,0] == 3:
                # P1 quad
                self.nnodel[ielem] = 4
                self.inpoel[ielem, :self.nnodel[ielem]] = allelems[i, 2+Mesh2d.ndtags:2+Mesh2d.ndtags+self.nnodel[ielem]]-1
                self.dtags[ielem, :Mesh2d.ndtags] = allelems[i, 2:2+Mesh2d.ndtags]
                ielem += 1
            elif allelems[i,0] == 10:
                # P2 quad
                self.nnodel[ielem] = 9
                self.inpoel[ielem, :self.nnodel[ielem]] = allelems[i, 2+Mesh2d.ndtags:2+Mesh2d.ndtags+self.nnodel[ielem]]-1
                self.dtags[ielem, :Mesh2d.ndtags] = allelems[i, 2:2+Mesh2d.ndtags]
                ielem += 1
            else:
                print("Mesh2d: readGmsh(): ! Invalid element type!")
        
        if ielem != self.nelem or iface != self.nbface:
            print("Mesh2d: readGmsh(): ! Error in adding up!")

        gc.collect()

# test
if __name__ == "__main__":
    fname = "try.msh"
    m = Mesh2d()
    m.readGmsh(fname)
    print(m.coords)
    print("---")
    print(m.inpoel)
    print("---")
    print(m.bface)
