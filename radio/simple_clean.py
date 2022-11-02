import numpy as np
from matplotlib import pyplot as plt
plt.ion()

def smooth_map(map,npix):
    nx=map.shape[0]
    ny=map.shape[1]
    x=np.arange(nx)
    x[x>nx//2]=x[x>nx//2]-nx
    y=np.arange(ny)
    y[y>ny//2]=y[y>ny//2]-ny
    rsqr=np.outer(x**2,np.ones(ny))+np.outer(np.ones(nx),y**2)
    kernel=np.exp(-0.5*rsqr/npix**2)
    #kernel=kernel/kernel.sum()
    mapft=np.fft.rfft2(map)
    kft=np.fft.rfft2(kernel)
    return np.fft.irfft2(mapft*kft)

class SrcCat:
    def __init__(self,x=None,y=None,flux=None):
        self.x=x
        self.y=y
        self.flux=x
        self.nsrc=0
        if not(x is None):
            self.nsrc=len(x)
            assert(len(x)==len(y))
            assert(len(x)==len(flux))
    def add_src(self,x,y,flux):
        if self.x is None:
            self.x=np.asarray([x],dtype='float')
            self.y=np.asarray([y],dtype='float')
            self.flux=np.asarray([flux],dtype='float')
        else:
            self.x=np.hstack([self.x,x])
            self.y=np.hstack([self.y,y])
            self.flux=np.hstack([self.flux,flux])
        self.nsrc=self.x.size
    def to_map(self,npix,pixsize,sig=None):
        map=np.zeros([npix,npix])
        for i in range(self.nsrc):
            x=self.x[i]/pixsize
            y=self.y[i]/pixsize
            ix=int(np.round(x))
            iy=int(np.round(y))
            map[ix,iy]=map[ix,iy]+self.flux[i]
        if not(sig is None):
            map=smooth_map(map,sig/pixsize)
        return map
class Vis:
    def __init__(self,u,v,vis=None,freq=1.4,ddish=None):
        self.vis=vis
        self.freq=freq
        self.lamda=2.9979e8/(1e9*freq)
        self.u=u/self.lamda
        self.v=v/self.lamda
        self.weights=np.ones(len(self.u))
        self.weights=self.weights/np.sum(self.weights)
        if ddish is None:
            self.sig=None
        else:
            #if we have a dish diameter, let's simplify primary
            #beam by assuming gaussian with sigma equal to half
            #of the distance to the first Airy null
            self.sig=1.22*self.lamda/ddish/2

    def fold(self):
        #make all u values >=0 so we can use
        #the real-to-real transforms
        mask=self.u<0
        self.u[mask]=-1*self.u[mask]
        self.v[mask]=-1*self.v[mask]
        if not(self.vis is None):
            self.vis[mask]=np.conj(self.vis[mask])
    
    def observe_cat(self,cat,do_pb=False):
        vis=np.zeros(len(self.u),dtype='complex')
        for i in range(cat.nsrc):
            mydot=self.u*cat.y[i]+self.v*cat.x[i]
            if do_pb==False:
                pb=1.0
            elif self.sig is None:
                pb=1.0
            else:
                d=np.sqrt(cat.y[i]**2+cat.x[i]**2)
                pb=np.exp(-0.5*d**2/self.sig**2)
            #print('pb and flux are ',pb,cat.flux[i],mydot.max())
            vis=vis+pb*cat.flux[i]*np.exp(-2J*np.pi*mydot)
            
        return vis
    def grid(self,npix,du,vis=None):
        if vis is None:
            vis=self.vis
        map=np.zeros([npix//2+1,npix],dtype='complex')
        ugrid=np.asarray(np.round(self.u/du),dtype='int')
        vgrid=np.asarray(np.round(self.v/du),dtype='int')
        nmax=npix//2        
        #print('lims are ',ugrid.min(),ugrid.max(),vgrid.min(),vgrid.max())
        isgood=(ugrid<nmax)
        isgood[vgrid>nmax]=False
        isgood[vgrid<-nmax]=False

        #print('isgood mean is ',np.mean(isgood))
        weights=self.weights
        if np.sum(isgood)<len(ugrid):
            print('due to requested map, keeping ',(np.sum(isgood)/len(ugrid)*100),' percent of the data.')
            ugrid=ugrid[isgood]
            vgrid=vgrid[isgood]
            vis=vis[isgood]
            weights=weights[isgood]
            weights=weights/np.sum(weights)
        nvis=len(vis)

        for i in range(nvis):
            map[ugrid[i],vgrid[i]]=map[ugrid[i],vgrid[i]]+weights[i]*vis[i]
        return map
    def dirty(self,npix,du,vis=None):
        uvmap=self.grid(npix,du,vis)
        dirty_map=(np.fft.irfft2(uvmap.T)).T*(npix**2/2)
        return dirty_map
def clean(vis,npix,pixsize,fac=0.5,niter=20):
    cat=SrcCat()
    cur_vis=vis.vis.copy()
    du=1/(npix*pixsize)
    for i in range(niter):
        #print('npix is ',npix,vis.u[0])
        map=vis.dirty(npix,du,cur_vis)
        plt.clf()
        plt.imshow(np.fft.fftshift(map))
        plt.colorbar()
        plt.show()
        plt.pause(0.01)
        ind=np.argmax(np.abs(map))
        ix=ind//npix
        if ix>npix//2:
            ix=ix-npix
        x=ix*pixsize
        iy=ind%npix
        if iy>npix//2:
            iy=iy-npix
        y=iy*pixsize

        flux=map[ix,iy]
        print('adding source at ',x,y,ix,iy,flux,np.max(np.abs(map)))
        if not(np.isfinite(flux)):
            assert(1==0)
        cat.add_src(y,x,fac*flux)
        cat_vis=vis.observe_cat(cat,do_pb=False)
        cur_vis=vis.vis-cat_vis
        #return map,cat_vis,cat

    return cur_vis,cat
        
    

uv=np.load('vla_uv_snap_d_array.npy')
vis=Vis(uv[:,0],uv[:,1],ddish=25)
vis.fold()
cat=SrcCat()
cat.add_src(0,0,1)
fac=np.pi/180/60 #convert radians to arcmin
nsrc=30
x=np.random.randn(nsrc)*nsrc*fac
y=np.random.randn(nsrc)*nsrc*fac
flux=1/np.random.rand(nsrc)
cat.add_src(x,y,flux)
#cat.add_src(0.01,0,10000)

myvis=vis.observe_cat(cat)

noise=1.0
myvis=myvis+noise*(np.random.randn(len(myvis))+1J*np.random.randn(len(myvis)))

vis.vis=myvis.copy()


npix=1024 #map size in pixels
pixsize=0.2*fac

du=1/(npix*pixsize)  #this is the u resolution of our requested map
uvmap=vis.grid(npix,du,myvis)
dirty_map=np.fft.fftshift(np.fft.irfft2(uvmap.T)).T*(npix**2/2)
beam=vis.dirty(npix,du,np.ones(len(vis.vis)))
nsmooth=0
while beam[nsmooth,0]>0.5:
    nsmooth=nsmooth+1
beam_sig=(2*nsmooth+1)/2.35

map2=np.fft.fftshift(vis.dirty(npix,du))
new_vis,new_cat=clean(vis,npix,pixsize,fac=0.5,niter=3*nsrc)

truth=smooth_map(cat.to_map(npix,pixsize),beam_sig)
cleaned=smooth_map(new_cat.to_map(npix,pixsize),beam_sig)+vis.dirty(npix,du,new_vis)


#m3=vis.dirty(npix,du,new_vis)
plt.figure(1)
plt.clf()
plt.imshow(np.fft.fftshift(truth))
plt.colorbar()
plt.title('Input Catalog')
plt.show()

plt.figure(2)
plt.clf()
plt.imshow(np.fft.fftshift(cleaned))
plt.title('Clean Map')
plt.colorbar()
plt.show()

plt.figure(3)
plt.clf()
plt.title('Dirty Map')
plt.imshow(np.fft.fftshift(vis.dirty(npix,du)))
plt.colorbar()
plt.show()

plt.figure(4)
plt.clf()
plt.title('Synthesized Beam')
plt.imshow(np.fft.fftshift(beam))
plt.colorbar()
plt.show()
