import numpy as np
from matplotlib import pyplot as plt
plt.ion()

t=np.linspace(0,0.2,8001)
vec=np.cos(120*np.pi*t)
vec2=np.sin(120*np.pi*t)
N=np.eye(len(t))*0.05
N=N+np.outer(vec,vec)
N=N+np.outer(vec2,vec2)

#generate correlated noise
L=np.linalg.cholesky(N)
mysim=L@np.random.randn(len(t))
plt.clf()
plt.plot(t,mysim)
plt.ylim([-3,3])
plt.show()

nu=60.5
signal=np.cos(t*2*np.pi*nu)
Ninv=np.linalg.inv(N)
lhs=signal.T@Ninv@signal #make A^T N^-1 A
myerr=1/np.sqrt(lhs)
print("error at ",nu," is ",myerr)

nuvec=np.linspace(55,65,101)
errvec=0*nuvec
for i in range(len(nuvec)):
    signal=np.cos(t*2*np.pi*nuvec[i])
    lhs=signal.T@Ninv@signal
    errvec[i]=1/np.sqrt(lhs)

