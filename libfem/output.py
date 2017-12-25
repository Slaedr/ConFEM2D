""" @file output.py
    @brief Routines for output to VTU file
"""

import numpy as np
from .mesh import Mesh2d

def writePointScalarToVTU(m, filename, scalarname, x):
    """ Writes one scalar mesh function into a VTU file.
    """
    fout = open(filename, 'w')
    fout.write("<VTKFile type=\"UnstructuredGrid\" version=\"0.1\" byte_order=\"LittleEndian\">\n")
    fout.write("<UnstructuredGrid>\n")
    fout.write("\t<Piece NumberOfPoints=\""+ str(m.npoin)+ "\" NumberOfCells=\""+ str(m.nelem)+ "\">\n")

    fout.write("\t\t<PointData Scalars=\""+scalarname+"\">\n")
    fout.write("\t\t\t<DataArray type=\"Float64\" Name=\""+ scalarname+ "\" Format=\"ascii\">\n")
    for i in range(m.npoin):
        fout.write("\t\t\t\t"+ str(x[i])+ "\n")
    fout.write("\t\t\t</DataArray>\n")

    fout.write("\t\t</PointData>\n")
    fout.write("\t\t<Points>\n")
    fout.write("\t\t<DataArray type=\"Float64\" NumberOfComponents=\"3\" Format=\"ascii\">\n")
    for ipoin in range(m.npoin):
        fout.write("\t\t\t"+ str(m.coords[ipoin,0])+ " "+ str(m.coords[ipoin,1])+ " 0.0\n")
    fout.write("\t\t</DataArray>\n\t\t</Points>\n")

    fout.write("\t\t<Cells>\n")
    fout.write("\t\t\t<DataArray type=\"UInt32\" Name=\"connectivity\" Format=\"ascii\">\n")
    for i in range(m.nelem):
        fout.write("\t\t\t\t")
        elemcode = 5
        if m.nnodel[i] == 4:
            elemcode = 9
        elif m.nnodel[i] == 6:
            elemcode = 22
        elif m.nnodel[i] == 9:
            elemcode = 28
        for j in range(m.nnodel[i]):
            fout.write(str(m.inpoel[i,j]) + " ")
        fout.write("\n")
    fout.write("\t\t\t</DataArray>\n")
    fout.write("\t\t\t<DataArray type=\"UInt32\" Name=\"offsets\" Format=\"ascii\">\n")
    for i in range(m.nelem):
        fout.write("\t\t\t\t" + str(m.nnodel[i]*(i+1)) + "\n")
    fout.write("\t\t\t</DataArray>\n")
    fout.write("\t\t\t<DataArray type=\"Int32\" Name=\"types\" Format=\"ascii\">\n")
    for i in range(m.nelem):
        fout.write("\t\t\t\t" + str(elemcode) + "\n")
    fout.write("\t\t\t</DataArray>\n")
    fout.write("\t\t</Cells>\n")

    fout.write("\t</Piece>\n</UnstructuredGrid>\n</VTKFile>")
    fout.close()
