import numpy as np

x=np.arange(2001)
sig=100
mycorr=np.exp(-0.5*x**2/sig**2)
mycorr=np.hstack([mycorr,np.flipud(mycorr[1:])])
myps=np.fft.rfft(mycorr)


dat=np.random.randn(len(mycorr))
datft=np.fft.rfft(dat)
datft=datft*np.sqrt(np.abs(myps))
dat_corr=np.fft.irfft(datft)

xx=np.arange(len(mycorr))
modsig=1
mymod=np.exp(-0.5*(xx-xx.mean())**2/modsig**2)
#make N^-1 A
mymodft=np.fft.rfft(mymod)
mymodft_filt=mymodft/myps
mymod_filt=np.fft.irfft(mymodft_filt)
