import numpy as np
from matplotlib import pyplot as plt

n=1000
sig=1

x=np.arange(n)
xmat=np.outer(x,np.ones(n))
dx=xmat-xmat.T
mycorr=np.exp(-0.5*dx**2/sig**2)
#add a bit of noise to diagonal for stability
mycorr=mycorr+0.000001*np.eye(n) 
L=np.linalg.cholesky(mycorr)
dat=L@np.random.randn(n)

plt.ion()
plt.clf()
plt.plot(dat)
plt.show()
plt.savefig('gauss_corrnoise_'+repr(sig)+'.png')

#let's make sure our simulated data looks like the correlation
nsim=10000
tmp=L@np.random.randn(n,nsim) #this lets us make many sims at once
corr2=tmp@tmp.T/nsim #average of the simulations
print('fractional correlation error is ',np.std(corr2-mycorr)/np.mean(mycorr))
