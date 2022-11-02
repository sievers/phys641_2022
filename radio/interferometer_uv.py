import numpy as np
from matplotlib import pyplot as plt
plt.ion()


lat=34.0784*np.pi/180

if False:
    antpos=np.loadtxt('vla_d_array.txt')
    du=2.0
    color='b.'
    array='d'
else:
    antpos=np.loadtxt('vla_a_array.txt')
    du=40.0
    color='k.'
    array='a'

antpos=antpos[:,:3]  #the last column is boring...
antpos=antpos*1e-9*3e8 #convert to meters, since the file is in ns
nant=antpos.shape[0]
nvis=nant*(nant-1)//2

#we can look at the antenna array in 2D coordinates by looking at the
#distance from the zenith.  The following math is useful for converting 3D
#positions to EW/NS coordinates on the Earth's surface
zenith=np.asarray([np.cos(lat),0,np.sin(lat)])
east=np.asarray([0,1,0])
north=np.cross(zenith,east)

mat=np.vstack([north,east,zenith])
xyz=antpos[:,:3]@mat.T

#we'll ignore the vertical correction and make the 2D UV coverage
uv=np.zeros([nvis,2])
icur=0
for i in range(nant):
    for j in range(i+1,nant):
        uv[icur,:]=xyz[i,:2]-xyz[j,:2]
        icur=icur+1
uv=np.vstack([uv,-1*uv]) #get the visibility conjugates

uv_3d=np.zeros([nvis,3])
icur=0
for i in range(nant):
    for j in range(i+1,nant):
        uv_3d[icur,:]=antpos[i,:]-antpos[j,:]
        icur=icur+1
uv_3d=np.vstack([uv_3d,-uv_3d])

#we'll observe over some range.  We'll define
#t_range in hours, then convert to angle
t_range=np.linspace(-4,4,61)
theta_range=t_range*2*np.pi/24
plt.clf()

#let's also pick a declination to observe at
dec=00.0*np.pi/180
zenith=np.asarray([np.cos(dec),0,np.sin(dec)])
east=np.asarray([0,1,0])
north=np.cross(zenith,east)

proj_mat=np.vstack([east,north])
plt.figure(1)
plt.clf()

pad=4
sz=int(np.max(np.abs(uv_3d))/du)
uv_mat=np.zeros([pad*2*sz,2*pad*sz])
for theta in theta_range:
    rot_mat=np.zeros([3,3])
    rot_mat[0,0]=np.cos(theta)
    rot_mat[1,1]=np.cos(theta)
    rot_mat[2,2]=1.0
    rot_mat[0,1]=np.sin(theta)
    rot_mat[1,0]=-np.sin(theta)
    uv_rot=uv_3d@rot_mat
    uv_snap=uv_rot@proj_mat.T
    if np.abs(theta)<0.001:
        np.save('vla_uv_snap_'+array+'_array',uv_snap)
    plt.plot(uv_snap[:,0],uv_snap[:,1],color)
    uv_int=np.asarray(uv_snap/du,dtype='int')
    for i in range(uv_snap.shape[0]):
        uv_mat[uv_int[i,0],uv_int[i,1]]=uv_mat[uv_int[i,0],uv_int[i,1]]+1

beam=np.abs(np.fft.ifft2(uv_mat))
x0=beam.shape[0]//2
dx=100
plt.figure(2)
plt.clf()
plt.imshow(np.fft.fftshift(beam))
plt.xlim([x0-dx,x0+dx])
plt.ylim([x0-dx,x0+dx])


    
