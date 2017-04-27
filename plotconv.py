#! /usr/bin/env python3
import sys
import numpy as np
from matplotlib import pyplot as plt

if(len(sys.argv) < 2):
	print("Error. Please provide input file name.")
	sys.exit(-1)
	
fname = sys.argv[1]
title = fname.split('/')[-1]

data = np.genfromtxt(fname)
n = data.shape[0]

pslope = np.zeros(data.shape[1])
labels = ['L2: ','H1: ']
symbs = ['o-', 's-']

for j in range(1,data.shape[1]):
	psigy = data[:,j].sum()
	sigx = data[:,0].sum()
	sigx2 = (data[:,0]*data[:,0]).sum()
	psigxy = (data[:,j]*data[:,0]).sum()

	pslope[j] = (n*psigxy-sigx*psigy)/(n*sigx2-sigx**2)
	print("Slope is " + str(pslope[j]))
	plt.plot(data[:,0],data[:,j],symbs[j-1],label=labels[j-1]+str(pslope[j]))


#plt.plot(data[:,0],data[:,2],'s-',label=labels[1]+str(pslope[1]))
plt.title("Grid-refinement (legend: slopes)") # + title)
plt.xlabel("Log mesh size")
plt.ylabel("Log error")
plt.legend()
plt.show()
