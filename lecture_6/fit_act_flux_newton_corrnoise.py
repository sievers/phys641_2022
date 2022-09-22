import numpy as np
from astropy.io import fits
from matplotlib import pyplot as plt


def pad_image(map):
    #return a double-sized map with images reflected up/down and left/right
    #in order to minimize FFT edge artifacts

    mm=np.hstack([map,np.fliplr(map)])
    mm=np.vstack([mm,np.flipud(mm)])
    return mm

def filter_image(patch,noiseft):
    patchft=np.fft.fft2(patch)
    patchft=patchft/noiseft
    patch_filt=np.real(np.fft.ifft2(patchft))
    return patch_filt

def mygauss(pars,width):
    vec=np.asarray(np.arange(-width,width),dtype='float')
    amp=pars[0]
    dx=pars[1]
    dy=pars[2]
    sig=pars[3]

    xvec=vec-dx
    yvec=vec-dy
    xmat=np.outer(xvec,np.ones(len(xvec)))
    ymat=np.outer(np.ones(len(yvec)),yvec)
    rmat=xmat**2+ymat**2
    model=np.exp(-0.5*(rmat/sig**2))*amp

    return model

def get_model_derivs(fun,pars,dpar,width):
    model=fun(pars,width)
    npar=len(pars)
    derivs=[None]*npar
    for i in range(npar):
        pp=pars.copy()
        pp[i]=pars[i]+dpar[i]
        m_plus=fun(pp,width)
        pp[i]=pars[i]-dpar[i]
        m_minus=fun(pp,width)
        derivs[i]=(m_plus-m_minus)/(2*dpar[i])
    return model,derivs

def get_model_derivs_ravel(fun,pars,dpar,width):
    model,derivs=get_model_derivs(fun,pars,dpar,width)
    model=np.ravel(model)
    npar=len(pars)
    derivs_out=np.empty([len(model),len(pars)])
    for i in range(npar):
        derivs_out[:,i]=np.ravel(derivs[i])
    return model,derivs_out

def newton(pars,data,fun,width,dpar,niter=10):
    for i in range(niter):
        model,derivs=get_model_derivs_ravel(fun,pars,dpar,width)
        resid=data-model
        lhs=derivs.T@derivs
        rhs=derivs.T@resid
        shift=np.linalg.inv(lhs)@rhs
        print('parameter shifts are ',shift)
        pars=pars+shift
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
width=50
patch=map[x0-width:x0+width,y0-width:y0+width]
#look at the raw data
plt.ion()
plt.figure(1)
plt.clf()
plt.imshow(patch)
plt.colorbar()
plt.show()
plt.savefig('patch_raw.png')

sig=1.0  #Gaussian sigma
xshift=0.0 #sub-pixel x-shift guess
yshift=0.0 #sub-pixel y-shift guess

dx=np.arange(-width,width)
dxmat=np.outer(dx,np.ones(len(dx)))
dymat=dxmat.T

drsqr=(dxmat-xshift)**2+(dymat-yshift)**2
mymodel=np.exp(-0.5*drsqr/sig**2)
#the least-squares best-fit amplitude for constant noise
amp=np.sum(mymodel*patch)/np.sum(mymodel**2)

resid=patch-mymodel*amp


plt.figure(2)
plt.clf()
plt.imshow(resid)
plt.colorbar()
plt.show()
plt.savefig('patch_modsub.png')

n=np.std(resid)
chi1=np.sum(patch**2/n**2)
chi2=np.sum(resid**2/n**2)
dchi=chi1-chi2
print('chisq improvement is ',dchi)

pars=np.asarray([3000,0,0,1],dtype='float')
dpar=np.asarray([1.0,0.01,0.01,0.01])/10

fitp=newton(pars,np.ravel(patch),mygauss,width,dpar)
mymod=mygauss(fitp,width)
fit_chisq=np.sum((patch-mymod)**2/n**2)
print('best-fit improvement is ',chi1-fit_chisq)
print('best-fit amplitude is ',fitp[0])
plt.clf()
plt.imshow(patch-mymod)
plt.colorbar()
plt.show()
plt.savefig('act_patch_modfit.png')



#now that we have a shape profile, let's re-estimate it's amplitude/error
#using a proper correlated noise matrix.  We'll leave in the option to use
#a white noise matrix as a check of our normalization factors.
if True:
    patch_clean=patch-mymod #get rid of signal
    patch_use=patch #this is what we'll fit the signal to
else:
    #give us the alternative of assuming white noise
    patch_clean=np.random.randn(patch.shape[0],patch.shape[1])
    patch_use=patch_clean+mymod #make a map with white noise only
patch_padded=pad_image(patch_clean)
patchft=np.fft.fft2(patch_padded)
noiseft=np.abs(patchft)**2
npad=5 #number of times to smooth out the noise
for i in range(npad):
    noiseft=noiseft+np.roll(noiseft,1,0)+np.roll(noiseft,-1,0)+np.roll(noiseft,1,1)+np.roll(noiseft,-1,1)
    noiseft=noiseft/5
noiseft=noiseft/patch_padded.size



mod_padded=pad_image(mymod)
mod_filt=filter_image(mod_padded,noiseft)

lhs=np.sum(mod_filt*mod_padded)/4
rhs=np.sum(mod_filt*pad_image(patch_use))/4

#as a sanity check, we'll make A^T (N^-1 d) instead of
#(N^-1A)^T d.  They ought to be identical, unless we messed up.
patch_filt=filter_image(pad_image(patch_use),noiseft)
rhs2=np.sum(mod_padded*patch_filt)/4
print('rhs two ways is ',rhs,rhs2) #if these disagree, we done goofed.

amp=rhs/lhs
err=1/np.sqrt(lhs)
lhs_white=np.sum(mymod**2)/np.mean(patch_clean**2)
err_white=1/np.sqrt(lhs_white)

print('amp/errs are ',amp,err,err_white,err/err_white)
print('in temperature: ',amp*fitp[0],err*fitp[0])
