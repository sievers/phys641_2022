import numpy as np
import zernike
from matplotlib import pyplot as plt
plt.ion()

npix=1024
xmax=2.5

x=np.arange(npix)
x[npix//2:]=x[npix//2:]-npix
x=x*1.0*xmax/npix
xmat=np.repeat([x],npix,axis=0)
ymat=xmat.transpose()
rmat=np.sqrt(xmat**2+ymat**2)
thmat=np.arctan2(xmat,ymat)

myzer,znvec=zernike.all_zernike(4,rmat,thmat)
plt.clf();
nx=3
ny=5
for i in range(nx):
    for j in range(ny):
        myind=i*ny+j
        plt.subplot(ny,nx,myind+1)
        plt.imshow(np.fft.fftshift(myzer[myind,:,:]))
        plt.axis('off')
plt.savefig('all_zernikes.png')
