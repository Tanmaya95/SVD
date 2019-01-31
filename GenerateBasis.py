import numpy as np
from scipy.sparse.linalg import svds
from matplotlib import pyplot as plt
import pylab as plt
import pycbc.waveform as pw
from pycbc.waveform import get_fd_waveform

N=65537
BG = [[0 for x in range(N)] for y in range(30)] 
for i in range (30):
    frequency=30.0
    sg, _ = pw.get_sgburst_waveform(approximant='TaylorF2', q=5.0, frequency=30.0+i*5.0, 
                                    delta_t=1./4096, hrss=3e-25, amplutude=1)
    sg.resize(N*2-2)                                
    sg.data = np.fft.fft(sg.data)#sg.to_frequencyseries()
    
    tran=sg.data[0:N]
    BG[i]=(tran.real)
    
    
#Y=np.loadtxt('ForSVD2.txt')

#print(Y)

np.savetxt('SGvectorsForSVD.txt',BG)

plt.plot(BG[5])
plt.show()

sg, _ = pw.get_sgburst_waveform(approximant='TaylorF2', q=5.0, frequency=40.0, 
                                    delta_t=1./4096, hrss=3e-25, amplutude=1)
sg.resize(N*2-2)                                
sg.data = np.fft.fft(sg.data)#sg.to_frequencyseries()

tran=sg.data[0:N]
tran=tran.real

np.savetxt('SG40.txt',tran)
sg, _ = pw.get_sgburst_waveform(approximant='TaylorF2', q=5.0, frequency=80.0, 
                                    delta_t=1./4096, hrss=3e-25, amplutude=1)
sg.resize(N*2-2)                                
sg.data = np.fft.fft(sg.data)#sg.to_frequencyseries()

tran=sg.data[0:N]
tran=tran.real

np.savetxt('SG80.txt',tran)
sg, _ = pw.get_sgburst_waveform(approximant='TaylorF2', q=5.0, frequency=30.0, 
                                    delta_t=1./4096, hrss=3e-25, amplutude=1)
sg.resize(N*2-2)                                
sg.data = np.fft.fft(sg.data)#sg.to_frequencyseries()

tran=sg.data[0:N]
tran=tran.real

np.savetxt('SG30.txt',tran)
