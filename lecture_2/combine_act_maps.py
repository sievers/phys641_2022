import numpy as np
from astropy.io import fits
from matplotlib import pyplot as plt

hdul=fits.open('act_dr4.01_s13_D6_pa1_f150_nohwp_night_3pass_4way_coadd_map_srcfree.fits')
#you can see what's in the file with e.g. print(hdul.info())
map_nosrc=hdul[0].data
hdul.close()

hdul=fits.open("act_dr4.01_s13_D6_pa1_f150_nohwp_night_3pass_4way_coadd_srcs.fits")
map_src=hdul[0].data
hdr=hdul[0].header
hdul.close()
map=map_nosrc+map_src

map_tt=np.asarray(map[0,:,:],dtype='float32')

plt.ion()
plt.clf()
plt.imshow(map_tt,vmin=-1000,vmax=1000)
plt.show()

hdu = fits.PrimaryHDU(map_tt)
hdul = fits.HDUList([hdu])
hdul.writeto('act_tt_map.fits')
