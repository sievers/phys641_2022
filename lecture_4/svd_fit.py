import numpy as np

x=np.linspace(-2,2)
y=np.cos(x)+np.sin(x)

ord=5
n=len(x)
mat=np.zeros([n,ord+1])
for i in range(ord+1):
    mat[:,i]=x**i

u,s,v=np.linalg.svd(mat,0)
fitp=v.T@np.diag(1/s)@u.T@y
pred=mat@fitp
print('reconstruction error RMS is ',np.std(pred-y))
print('inverse condition number of SVD is ',s.min()/s.max())

lhs=mat.T@mat
e,v=np.linalg.eigh(lhs)
print('inverse condition number of A^TA is ',e.min()/e.max())
