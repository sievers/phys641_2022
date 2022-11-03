import numpy as np
from matplotlib import pyplot as plt
import healpy
#let's make a plot of individual Ylm's
l=2
m=0

nlm=(l+1)*(l+2)//2  #where does this come from?

#healpix indexing is such that all the m=0 modes are 
#first, m=1 modes are next, etc. use this bit to brute-force
#find where in healpix ordering our requested mode will 
icur=0
for mm in range(m):
    icur=icur+(l+1-mm)
icur=icur+(l-m)

alm=np.zeros(nlm,dtype='complex')
alm[icur]=1.0
nside=256
map=healpy.alm2map(alm,nside)
alm_back=healpy.map2alm(map,lmax=l,iter=1)
plt.ion()
healpy.mollview(map)
plt.title('l='+repr(l)+', m='+repr(m))

