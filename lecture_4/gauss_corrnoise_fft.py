import numpy as np
from matplotlib import pyplot as plt

n=1000
x=np.fft.fftfreq(n)*n
sig=100
mycorr=np.exp(-0.5*x**2/sig**2)
myps=np.fft.rfft(mycorr)

dat=np.random.randn(n)
datft=np.fft.rfft(dat)
dat_corr=np.fft.irfft(datft*np.sqrt(myps))
print(np.sqrt(np.mean(dat_corr**2)))
plt.ion()
plt.clf()
plt.plot(dat_corr)
plt.show()
plt.savefig('gauss_corrnoise_fft_' + repr(sig)+'.png')
