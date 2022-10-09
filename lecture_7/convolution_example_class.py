import numpy as np
from matplotlib import pyplot as plt
plt.ion()

n=100000

x=np.zeros(n)

t=np.arange(n)
width=100
r=np.exp(-t/width)
plt.clf()
plt.plot(r)
plt.show()


cr=np.random.rand(n)**-1

crft=np.fft.rfft(cr)
rft=np.fft.rfft(r)
signal=np.fft.irfft(crft*rft)
plt.clf()
plt.plot(signal)
plt.show()
