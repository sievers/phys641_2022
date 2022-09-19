import numpy as np
from matplotlib import pyplot as plt
plt.ion()

npt=2001
t=np.linspace(0,5/60,npt)
N=0.01*np.eye(npt)
mycos=np.cos(120*np.pi*t)
N=N+np.outer(mycos,mycos)
mysin=np.sin(120*np.pi*t)
N=N+np.outer(mysin,mysin)
L=np.linalg.cholesky(N)
d=L@np.random.randn(npt)
plt.clf()
plt.plot(t,d)
plt.show()

Ninv=np.linalg.inv(N)
nuvec=np.linspace(35,95,201)
myerrs=0.0*nuvec #this will be the correct error bar
myerrs2=0*myerrs #this will be the error bar ignoring our correlated noise
Ninv2=np.linalg.inv(np.diag(np.diag(N)))
for i in range(len(nuvec)):
    nu=nuvec[i]
    phi=2*np.pi*np.random.rand(1)
    myvec=np.cos(2*np.pi*nu*t+phi)
    lhs=myvec.T@Ninv@myvec
    myerrs[i]=np.sqrt(1/lhs)
    lhs2=myvec.T@Ninv2@myvec
    myerrs2[i]=np.sqrt(1/lhs2)
    
plt.clf()
plt.plot(myerrs2/myerrs) #plot the ratio of the bad to correct errors
plt.show()
plt.savefig('error_ratio.png')
