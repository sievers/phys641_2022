import numpy as np
from matplotlib import pyplot as plt
N=1000
x=np.linspace(-5,5,N)
y=np.exp(-0.5*x**2/2*100)

F=np.fft.fft(y)

k=np.arange(N)
delta=1024
phase=np.exp(2*np.pi*1J*k*delta/N)

y_shift=np.real(np.fft.ifft(F*phase))

plt.ion()
plt.clf()
plt.plot(y)
plt.plot(y_shift)
plt.show()
