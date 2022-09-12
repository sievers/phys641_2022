import numpy as np
from astropy.io import fits
from matplotlib import pyplot as plt

#I made this map for you
#look at combine_act_maps.py if you want to see more
#FITS tricks
hdul=fits.open("act_tt_map.fits")
map=hdul[0].data
hdul.close()
map=np.asarray(map,dtype='float')

#we can look at the map, and see there's a bright source here
x0=355
y0=1767


#let's snip a bit of the map out around the bright source
width=50
patch=map[x0-width:x0+width,y0-width:y0+width]
#look at the raw data
plt.ion()
plt.figure(1)
plt.clf()
plt.imshow(patch)
plt.colorbar()
plt.show()
plt.savefig('patch_raw.png')

sig=1.2  #Gaussian sigma
xshift=0.25 #sub-pixel x-shift guess
yshift=-0.3 #sub-pixel y-shift guess

dx=np.arange(-width,width)
dxmat=np.outer(dx,np.ones(len(dx)))
dymat=dxmat.T

drsqr=(dxmat-xshift)**2+(dymat-yshift)**2
mymodel=np.exp(-0.5*drsqr/sig**2)
#the least-squares best-fit amplitude for constant noise
amp=np.sum(mymodel*patch)/np.sum(mymodel**2)

resid=patch-mymodel*amp


plt.figure(2)
plt.clf()
plt.imshow(resid)
plt.colorbar()
plt.show()
plt.savefig('patch_modsub.png')

sigma=np.std(resid)
chi1=np.sum(patch**2/sigma**2)
chi2=np.sum(resid**2/sigma**2)
dchi=chi1-chi2
print('chisq improvement is ',dchi)

