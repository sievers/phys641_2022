import numpy as np


x=np.linspace(-np.pi,np.pi,1024)
win=(np.cos(x)+1)/2

F=np.fft.rfft(x)
plt.clf();plt.plot(np.abs(F));plt.show()

F2=np.fft.rfft(x*win)
plt.plot(F2)
