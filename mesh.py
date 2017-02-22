""" @brief 2D Mesh handling class
"""

import gc
import numpy as np
from numba import jit, jitclass, int64, int32, float64

meshclassspec = [('nbtags',int64),('ndtags',int64), ('npoin', int64), ('nelem',int64), ('nbface',int64), ('maxnnodel',int64), ('nnodel',int32[:]), ('nnofa',int32[:]),
        ('coords', float64[:,:]), ('inpoel', int32[:,:]), ('bface',int32[:,:]), ('dtags',int32[:,:]) ]

@jitclass(meshclassspec)
class Mesh2d:
    """ @brief Stores the mesh data.

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

    Note that BCs are handled later with the help of the tags stored in bface, which are read from the Gmsh file.
    """
    def __init__(self, npo, ne, nf, maxnp, nbt, ndt, _coords, _inpoel, _bface, _nnodel, _nnofa, _dtags):
        self.npoin = npo
        self.nelem = ne
        self.nbface = nf
        self.maxnnodel = maxnp
        self.nbtags = nbt
        self.ndtags = ndt
        self.coords = np.zeros((self.npoin,2),dtype=np.float64)
        self.inpoel = np.zeros((self.nelem,self.maxnnodel),dtype=np.int32)
        self.bface = np.zeros((self.nbface,2+self.nbtags),dtype=np.int32)
        self.nnofa = np.zeros(self.nbface, dtype=np.int32)
        self.nnodel = np.zeros(self.nelem,dtype=np.int32)
        self.dtags = np.zeros((self.nelem,2),dtype=np.int32)
        self.coords[:,:] = _coords[:,:]
        self.inpoel[:,:] = _inpoel[:,:]
        self.bface[:,:] = _bface[:,:]
        self.nnodel[:] = _nnodel[:]
        self.nnofa[:] = _nnofa[:]
        self.dtags[:,:] = _dtags[:,:]

class Mesh2dIO:
    """ @brief Reads and processes the mesh data.

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

    Note that BCs are handled later with the help of the tags stored in bface, which are read from the Gmsh file.
    """
    def __init__(self):
        self.npoin = 0
        self.nelem = 0
        self.nbface = 0
        self.maxnnodel = 4
        self.nbtags = 2
        self.ndtags = 2
        """self.coords = np.zeros((2,2),dtype=np.float64)
        self.inpoel = np.zeros((2,2),dtype=np.int32)
        self.bface = np.zeros((2,2),dtype=np.int32)
        self.nnofa = np.zeros(2, dtype=np.int32)
        self.nnodel = np.zeros(2,dtype=np.int32)
        self.dtags = np.zeros((2,2),dtype=np.int32)"""

    def readGmsh(self, fname):
        """ Reads a Gmsh2 mesh file."""
        f = open(fname,'r')
        for i in range(4):
            f.readline()
        
        self.npoin = int(f.readline())
        print("readGmsh(): Num points = " + str(self.npoin))
        temp = np.fromfile(f, dtype=float, count=self.npoin*4, sep=" ").reshape((self.npoin,4))
        self.coords = temp[:,1:-1]
        print("readGmsh(): Coords read. Shape of coords is "+str(self.coords.shape))

        for i in range(2):
            f.readline()
        nallelem = int(f.readline())
        allelems = np.zeros((nallelem,self.ndtags+12))
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
                print("! readGmsh(): ! Invalid element type!")
            for j in range(len(elem)):
                allelems[i,j] = elem[j]
        f.close()

        self.bface = np.zeros((self.nbface,2+self.nbtags),dtype=int)
        self.inpoel = np.zeros((self.nelem, self.maxnnodel), dtype=int)
        self.dtags = np.zeros((self.nelem,self.ndtags), dtype=int)
        self.nnodel = np.zeros(self.nelem,dtype=int)
        self.nnofa = np.zeros(self.nbface,dtype=int)
        iface = 0; ielem = 0

        for i in range(nallelem):
            if allelems[i,0] == 1:
                # P1 line segment
                self.nnofa[iface] = 2
                self.bface[iface, :self.nnofa[iface]] = allelems[i, 2+self.nbtags:2+self.nbtags+self.nnofa[iface]]-1
                self.bface[iface, self.nnofa[iface]:self.nnofa[iface]+self.nbtags] = allelems[i, 2:2+self.nbtags]
                iface += 1
            elif allelems[i,0] == 8:
                # P2 line segment
                self.nnofa[iface] = 3
                self.bface[iface, :self.nnofa[iface]] = allelems[i, 2+self.nbtags:2+self.nbtags+self.nnofa[iface]]-1
                self.bface[iface, self.nnofa[iface]:self.nnofa[iface]+self.nbtags] = allelems[i, 2:2+self.nbtags]
                iface += 1
            elif allelems[i,0] == 2:
                # P1 tri
                self.nnodel[ielem] = 3
                self.inpoel[ielem, :self.nnodel[ielem]] = allelems[i, 2+self.ndtags:2+self.ndtags+self.nnodel[ielem]]-1
                self.dtags[ielem, :self.ndtags] = allelems[i, 2:2+self.ndtags]
                ielem += 1
            elif allelems[i,0] == 9:
                # P2 tri
                self.nnodel[ielem] = 6
                self.inpoel[ielem, :self.nnodel[ielem]] = allelems[i, 2+self.ndtags:2+self.ndtags+self.nnodel[ielem]]-1
                self.dtags[ielem, :self.ndtags] = allelems[i, 2:2+self.ndtags]
                ielem += 1
            elif allelems[i,0] == 3:
                # P1 quad
                self.nnodel[ielem] = 4
                self.inpoel[ielem, :self.nnodel[ielem]] = allelems[i, 2+self.ndtags:2+self.ndtags+self.nnodel[ielem]]-1
                self.dtags[ielem, :self.ndtags] = allelems[i, 2:2+self.ndtags]
                ielem += 1
            elif allelems[i,0] == 10:
                # P2 quad
                self.nnodel[ielem] = 9
                self.inpoel[ielem, :self.nnodel[ielem]] = allelems[i, 2+self.ndtags:2+self.ndtags+self.nnodel[ielem]]-1
                self.dtags[ielem, :self.ndtags] = allelems[i, 2:2+self.ndtags]
                ielem += 1
            else:
                print("! readGmsh(): ! Invalid element type!")
        
        if ielem != self.nelem or iface != self.nbface:
            print("Mesh2d: readGmsh(): ! Error in adding up!")


if __name__ == "__main__":
    fname = "try.msh"
    m = Mesh2d()
    self.readGmsh(fname)
    print(m.coords)
    print("---")
    print(m.inpoel)
    print("---")
    print(m.bface)
