# -*- coding: utf-8 -*-
"""
Created on Sun Nov 18 00:29:40 2018

@author: tanma
"""

import numpy as np
from scipy.sparse.linalg import svds
from matplotlib import pyplot as plt
import pylab as plt

Pfre=np.loadtxt('Pfre.txt')
phen=np.loadtxt('IMRP.txt')
SG=np.loadtxt('SG40.txt')
Y=np.loadtxt('SVDbasis.txt')
'''
print(Y)
print(Y.shape)
print(min(Y.shape))
print(Pfre)
print(SG)
print(phen)
print(len(phen))
print(len(SG))
'''
#plt.figure()
#plt.plot(Pfre,Y[15])
#plt.show()
print(len(Pfre))
#plt.figure()
#plt.plot(Pfre,SG)
#plt.show()
phen[0]=0
print(np.dot(phen,phen))
PP=np.sqrt(np.dot(phen,phen))
print(PP)
CBC=phen/PP

noise1=np.random.random(len(Pfre))-0.5
#phen=phen+noise1/1.5
plt.figure()
plt.plot(Pfre,phen)
plt.plot(Pfre,CBC)
plt.show()
SGm=np.sqrt(np.dot(SG,SG))
print(SGm)
NSG=SG/SGm
noise=np.random.random(len(Pfre))-0.5

SGF=np.fft.fft(SG)
plt.figure()

#plt.plot(Pfre,SG)
#plt.plot(Pfre,NSG)
plt.plot(Pfre,Y[10])
plt.show()
print(np.dot(Y[10],Y[10]))
print(np.dot(Y[10],phen))
print(np.dot(Y[10],CBC))
print(np.dot(Y[10],SG))
print(np.dot(Y[10],NSG))

#SG=SG+noise/1.5
#plt.figure()
#plt.plot(Pfre,SG)
#plt.show()
print(np.dot(phen,Y[12]))
print(np.dot(NSG,Y[4]))

plt.figure()
plt.title('Comparison of Waveforms')
plt.xlabel('Frequency')
plt.ylabel('Normalised Waveform')
plt.plot(Pfre,CBC)
plt.plot(Pfre,NSG)
plt.plot(Pfre,Y[18])
plt.ylim(-0.15,0.15)
plt.xlim(0,160)
plt.legend(('IMRPhenomD','Sine-Gaussian Glitch','SVD Sine-Gaussian basis'),loc='upper right')
plt.show()

Z=[]
m=0
mp=0
normSG=[]
normphen=[]
norm=0
norm_nor=0

for i in range(min(Y.shape)):
    mp=np.dot(Y[i],CBC)
    m=np.dot(Y[i],NSG)
    normSG.append(m*m)
    normphen.append(mp*mp)
    norm=norm+m
    norm_nor=norm_nor+mp

print(normSG)

plt.figure()
plt.plot(normSG)

plt.show()

plt.figure()
plt.plot(normphen)

plt.show()

print(normphen)
plt.figure()
plt.plot(normSG)
plt.plot(normphen)
plt.title('Overlap with sine-Gaussian basis',fontsize=20)
plt.xlabel('Basis vector',fontsize=18)
plt.ylabel('Overlap (normalised dot product)',fontsize=18)
plt.grid(True)
plt.ylim(-0.025,1.0)
plt.legend(('noisy sine-Gaussian', 'noisy GW signal'),loc='upper right')
plt.show()
print(norm)
print(norm_nor)
'''
#redacted sine gaussian svd basis
RSG=[]
for i in range(min(Y.shape)):
    if i!=20 and i!=21 and i!=22:
        RSG.append(Y[i])

normSG1=[]
normphen1=[]
norm=0
norm_nor=0
m=0
mp=0
for i in range(26):
    m=np.dot(RSG[i],SG)
    mp=np.dot(RSG[i],phen)
    normSG1.append(m*m)
    normphen1.append(mp*mp)
    norm=norm+m
    norm_nor=norm_nor+mp

print(normSG1)


plt.figure()
plt.plot(normSG1)

print(normphen1)

plt.plot(normphen1)
plt.title('Overlap with Redacted sine-Gaussian basis',fontsize=20)
plt.xlabel('Redacted basis vector',fontsize=18)
plt.ylabel('Overlap (normalised dot product)',fontsize=18)
plt.grid(True)
plt.ylim(-0.025,1.0)
plt.legend(('noisy sine-Gaussian', 'noisy GW signal'),loc='upper right')
plt.show()
print(norm)

print(norm_nor)
'''
