import numpy as np
import zernike
from matplotlib import pyplot as plt
plt.ion()

npix=2048
xmax=20

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
pad=2
for i in range(nx):
    for j in range(ny):
        myind=i*ny+j
        mybeam=np.fft.fft2(np.exp(2*np.pi*1J*myzer[myind,:,:])*myzer[0,:,:])
        mybeam=np.abs(mybeam)**2        
        plt.subplot(ny,nx,myind+1)
        plt.imshow(np.fft.fftshift(mybeam))
        plt.axis([npix//2-pad*xmax,npix//2+pad*xmax,npix//2-pad*xmax,npix//2+pad*xmax])
        plt.axis('off')
plt.savefig('aberrations.png')
