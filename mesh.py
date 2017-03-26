""" @brief 2D Mesh handling
"""

import numpy as np
import numba
from numba import jit, jitclass, int64, float64

meshclassspec = [('nbtags',int64),('ndtags',int64), ('npoin', int64), ('nelem',int64), ('nbface',int64), ('maxnnodel',int64), ('maxnnofa',int64), 
        ('nnodel',int64[:]), ('nnofa',int64[:]), ('nfael', int64[:]),
        ('coords', float64[:,:]), ('inpoel', int64[:,:]), ('bface',int64[:,:]), ('dtags',int64[:,:]), ('h',float64) ]

#@jitclass(meshclassspec)
class Mesh2d:
    """ @brief Stores the mesh data in a jit-able form

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
    def __init__(self, npo, ne, nf, maxnp, maxnf, nbt, ndt, _coords, _inpoel, _bface, _nnodel, _nfael, _nnofa, _dtags):
        self.npoin = npo
        self.nelem = ne
        self.nbface = nf
        self.maxnnodel = maxnp
        self.maxnnofa = maxnf
        self.nbtags = nbt
        self.ndtags = ndt
        self.coords = np.zeros((self.npoin,2),dtype=np.float64)
        self.inpoel = np.zeros((self.nelem,self.maxnnodel),dtype=np.int64)
        self.bface = np.zeros((self.nbface,self.maxnnofa+self.nbtags),dtype=np.int64)
        self.nnofa = np.zeros(self.nbface, dtype=np.int64)
        self.nnodel = np.zeros(self.nelem,dtype=np.int64)
        self.nfael = np.zeros(self.nelem,dtype=np.int64)
        self.dtags = np.zeros((self.nelem,2),dtype=np.int64)
        self.coords[:,:] = _coords[:,:]
        self.inpoel[:,:] = _inpoel[:,:]
        self.bface[:,:] = _bface[:,:]
        self.nnodel[:] = _nnodel[:]
        self.nfael[:] = _nfael[:]
        self.nnofa[:] = _nnofa[:]
        self.dtags[:,:] = _dtags[:,:]

        # compute mesh size parameter h 
        # length of longest edge in the mesh - reasonable for triangular elements
        self.h = 0.0
        for ielem in range(self.nelem):
            localh = 0.0
            for iface in range(self.nfael[ielem]):
                facevec = self.coords[self.inpoel[ielem, (iface+1) % self.nfael[ielem]],:] - self.coords[self.inpoel[ielem,iface],:]
                faceh = np.sqrt(facevec[0]*facevec[0]+facevec[1]*facevec[1])
                if self.h < faceh:
                    self.h = faceh
        
        #print("Mesh2d: Stored mesh data. nelem = "+str(self.nelem)+", npoin = "+str(self.npoin)+", nbface = "+str(self.nbface))


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
        self.maxnnodel = 6
        self.maxnnofa = 3
        self.nbtags = 2
        self.ndtags = 2

    def readGmsh(self, fname):
        """ Reads a Gmsh2 mesh file."""
        f = open(fname,'r')
        for i in range(4):
            f.readline()
        
        self.npoin = int(f.readline())
        temp = np.fromfile(f, dtype=float, count=self.npoin*4, sep=" ").reshape((self.npoin,4))
        self.coords = temp[:,1:-1]
        print("  readGmsh(): Coords read. Shape of coords is "+str(self.coords.shape))

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

        self.bface = np.zeros((self.nbface, self.maxnnofa+self.nbtags),dtype=int)
        self.inpoel = np.zeros((self.nelem, self.maxnnodel), dtype=int)
        self.dtags = np.zeros((self.nelem,self.ndtags), dtype=int)
        self.nnodel = np.zeros(self.nelem,dtype=int)
        self.nfael = np.zeros(self.nelem,dtype=np.int32)
        self.nnofa = np.zeros(self.nbface,dtype=np.int32)
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
                self.nfael[ielem] = 3
                self.inpoel[ielem, :self.nnodel[ielem]] = allelems[i, 2+self.ndtags:2+self.ndtags+self.nnodel[ielem]]-1
                self.dtags[ielem, :self.ndtags] = allelems[i, 2:2+self.ndtags]
                ielem += 1
            elif allelems[i,0] == 9:
                # P2 tri
                self.nnodel[ielem] = 6
                self.nfael[ielem] = 3
                self.inpoel[ielem, :self.nnodel[ielem]] = allelems[i, 2+self.ndtags:2+self.ndtags+self.nnodel[ielem]]-1
                self.dtags[ielem, :self.ndtags] = allelems[i, 2:2+self.ndtags]
                ielem += 1
            elif allelems[i,0] == 3:
                # P1 quad
                self.nnodel[ielem] = 4
                self.nfael[ielem] = 4
                self.inpoel[ielem, :self.nnodel[ielem]] = allelems[i, 2+self.ndtags:2+self.ndtags+self.nnodel[ielem]]-1
                self.dtags[ielem, :self.ndtags] = allelems[i, 2:2+self.ndtags]
                ielem += 1
            elif allelems[i,0] == 10:
                # P2 quad
                self.nnodel[ielem] = 9
                self.nfael[ielem] = 4
                self.inpoel[ielem, :self.nnodel[ielem]] = allelems[i, 2+self.ndtags:2+self.ndtags+self.nnodel[ielem]]-1
                self.dtags[ielem, :self.ndtags] = allelems[i, 2:2+self.ndtags]
                ielem += 1
            else:
                print("! readGmsh(): ! Invalid element type!")
        
        if ielem != self.nelem or iface != self.nbface:
            print("Mesh2d: readGmsh(): ! Error in adding up!")

#@jit(nopython=True, cache=True)
def createMesh(npoin, nelem, nbface, maxnnodel, maxnnofa, nbtags, ndtags, coords, inpoel, bface, nnodel, nfael, nnofa, dtags):
    # Create a compiled Mesh2d object
    m = Mesh2d(npoin, nelem, nbface, maxnnodel, maxnnofa, nbtags, ndtags, coords, inpoel, bface, nnodel, nfael, nnofa, dtags)
    return m

if __name__ == "__main__":
    mio = Mesh2dIO()
    mio.readGmsh("../Meshes-and-geometries/squarehole0.msh")
    m = Mesh2d(mio.npoin, mio.nelem, mio.nbface, mio.maxnnodel, mio.maxnnofa, mio.nbtags, mio.ndtags, mio.coords, mio.inpoel, mio.bface, mio.nnodel, mio.nfael, mio.nnofa, mio.dtags)
    mio = 0
    print(numba.typeof(m))
