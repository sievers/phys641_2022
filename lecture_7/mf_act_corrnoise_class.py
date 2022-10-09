import numpy as np
from astropy.io import fits
from matplotlib import pyplot as plt
plt.ion()


def smooth_map(map,npass):
    tmp=map.copy()
    for i in range(npass):
        tmp=tmp+np.roll(map,1,0)+np.roll(map,-1,0)+np.roll(map,1,1)+np.roll(map,-1,1)
        tmp=tmp/5
    return tmp

def pad_map(map):
    map=np.hstack([map,np.fliplr(map)])
    map=np.vstack([map,np.flipud(map)])
    return map


def gauss2d(pars,x):
    x0=pars[0]
    y0=pars[1]
    amp=pars[2]
    sig=pars[3]
    
    dx=x-x0
    dy=x-y0
    
    dxmat=np.outer(dx,np.ones(len(dx)))
    dymat=np.outer(np.ones(len(dy)),dy)
    rsqr=dxmat**2+dymat**2
    map=amp*np.exp(-.5*rsqr/sig**2)
    return map

def get_derivs_ravel(fun,pars,dp,x):
    mymod=fun(pars,x)
    npar=len(pars)
    dplus=[None]*npar
    dminus=[None]*npar
    for i in range(npar):
        pp=pars.copy()
        pp[i]=pp[i]+dp[i]
        dplus[i]=fun(pp,x)
        pp=pars.copy()
        pp[i]=pp[i]-dp[i]
        dminus[i]=fun(pp,x)
    n=dplus[0].size
    A=np.empty([n,npar])
    #actually do the numerical derivatives
    for i in range(npar):
        dd=(dplus[i]-dminus[i])/(2*dp[i])
        A[:,i]=np.ravel(dd)
        
    return np.ravel(mymod),A

def newton(pars,fun,data,x,dp,niter=10):
    for i in range(niter):
        mod,A=get_derivs_ravel(fun,pars,dp,x)
        r=data-mod
        lhs=A.T@A
        rhs=A.T@r
        dp=np.linalg.inv(lhs)@rhs
        print('on interation ',i,' parameter shifts are ',dp)
        pars=pars+dp
    return pars
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
width=100
patch=map[x0-width:x0+width,y0-width:y0+width]
x=np.arange(0,patch.shape[0])

guess=np.asarray([width,width,patch.max(),1.0])
model=gauss2d(guess,x)
dp=np.asarray([0.01,0.01,1.0,0.01])
mod,derivs=get_derivs_ravel(gauss2d,guess,dp,x)

fitp=newton(guess,gauss2d,np.ravel(patch),x,dp)
modfit=gauss2d(fitp,x)

patch2=pad_map(patch-modfit)
patchft=np.fft.fft2(patch2)
myft=np.fft.fftshift(np.abs(patchft))[1:,1:]
patch_smooth=smooth_map(np.abs(patchft)**2,30)
Ninv=1/patch_smooth
modpad=pad_map(modfit)
modft=np.fft.fft2(modpad)
mod_filt=np.fft.ifft2(modft*Ninv)
dat_filt=np.fft.ifft2(np.fft.fft2(patch2)*Ninv)
lhs=np.sum(mod_filt*modpad)
rhs=np.sum(mod_filt*pad_map(patch))

#--------------------------------------------------------------------------------
#now do the matched filter
#--------------------------------------------------------------------------------
#first thing is to make a model/template for the signal we want to find

x=np.fft.fftfreq(patch2.shape[0])*patch2.shape[0]
y=np.fft.fftfreq(patch2.shape[1])*patch2.shape[1]
rsqr=np.outer(x**2,np.ones(len(y)))+np.outer(np.ones(len(x)),y**2)
template=np.exp(-0.5*rsqr/fitp[-1]**2)
#let's make N^-1 times the template
tft=np.fft.fft2(template)
datft=np.fft.fft2(pad_map(patch))
Ninvt=tft*Ninv  #still in Fourier space
mf_rhs=np.real(np.fft.irfft2(Ninvt*np.conj(datft)))
mf_rhs_check=np.real(np.fft.irfft2(np.conj(datft*Ninv)*tft))
#make A^T N^-1 A
NinvA_real=np.real(np.fft.ifft2(Ninvt))

