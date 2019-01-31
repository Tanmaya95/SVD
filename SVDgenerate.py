# -*- coding: utf-8 -*-
"""
Created on Wed Dec 26 13:03:48 2018

@author: tanma
"""

import numpy as np
from scipy.sparse.linalg import svds
from matplotlib import pyplot as plt
import pylab as plt

Pfre=np.loadtxt('Pfre.txt')
Y=np.loadtxt('SGvectorsForSVD.txt')
#Z=np.loadtxt('SVDbasisfinal22.txt')
#np.savetxt('SGpycbc08.txt',Y[8])
plt.figure()
plt.plot(Pfre,Y[8])
plt.show()
#plt.figure()
#plt.plot(Pfre,Z[8])
#plt.show()
print(Y.shape)
#print(Z.shape)

U, S, VT = svds(Y,k=29) #set k to be the number of basis vectors you want to get ... it has to be strictly less than the lowest dimension of Y

print(VT.shape)
plt.figure()
plt.plot(Pfre,VT[10])
plt.show()

np.savetxt('SVDbasis.txt',VT)

