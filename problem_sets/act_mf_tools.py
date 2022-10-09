import numpy as np
from astropy.io import fits
from matplotlib import pyplot as plt
from scipy import fft
import time

def pad_map(map):
    map=np.hstack([map,np.fliplr(map)])
    map=np.vstack([map,np.flipud(map)])
    return map

def get_gauss_kernel(map,sig,norm=False):
    nx=map.shape[0]
    x=np.fft.fftfreq(map.shape[0])*map.shape[0]
    y=np.fft.fftfreq(map.shape[1])*map.shape[1]
    rsqr=np.outer(x**2,np.ones(map.shape[1]))+np.outer(np.ones(map.shape[0]),y**2)
    kernel=np.exp((-0.5/sig**2)*rsqr)
    if norm:
        kernel=kernel/kernel.sum()
    return kernel




hdul=fits.open('advact_tt_patch.fits')
map=hdul[0].data
hdul.close()
map=np.asarray(map,dtype='float')
print('read map')


t1=time.time()
mapft=np.fft.fft2(map)
t2=time.time()
print("numpy time to fft is ",t2-t1)
#this is going to be faster than np.fft for the FFT routines
t1=time.time()
mapft=fft.fft2(map,workers=4)
t2=time.time()
print('elapsed scipy fft time is ',t2-t1)


#if you want to write a map out
#hdu = fits.PrimaryHDU(pad_smooth)
#hdul = fits.HDUList([hdu])
#hdul.writeto('advact_tt_patch_filt.fits',overwrite=True)


