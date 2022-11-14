import numpy as np
from astropy.io import fits
from matplotlib import pyplot as plt
import healpy
import camb
import time
plt.ion()


def get_spectrum(pars,lmax=3000):
    #print('pars are ',pars)
    H0=pars[0]
    ombh2=pars[1]
    omch2=pars[2]
    tau=pars[3]
    As=pars[4]
    ns=pars[5]
    pars=camb.CAMBparams()
    pars.set_cosmology(H0=H0,ombh2=ombh2,omch2=omch2,mnu=0.06,omk=0,tau=tau)
    pars.InitPower.set_params(As=As,ns=ns,r=0)
    pars.set_for_lmax(lmax,lens_potential_accuracy=0)
    results=camb.get_results(pars)
    powers=results.get_cmb_power_spectra(pars,CMB_unit='muK')
    cmb=powers['total']
    tt=cmb[:,0]
    return tt[2:]
def read_map(fname,nside=None):
    hdul=fits.open(fname)
    data=hdul[1].data
    hdul.close()

    map=data['I_STOKES']
    map=np.asarray(map,dtype='float32')
    if not(nside is None):
        map=healpy.ud_grade(map,nside)
    map=healpy.reorder(map,inp='NESTED',out='RING')
    return map


As=np.exp(3.044)*1e-10
pars=np.asarray([67.32,0.02238,0.12,0.0543,As,0.9649])
model=get_spectrum(pars)


nside=1024
if True:
    #you can get these maps, and others, from:
    #https://irsa.ipac.caltech.edu/data/Planck/release_3/all-sky-maps/matrix_cmb.html
    t1=time.time()
    m1=read_map('COM_CMB_IQU-nilc_2048_R3.00_hm1.fits')#,nside)
    m2=read_map('COM_CMB_IQU-nilc_2048_R3.00_hm2.fits')#,nside)
    map=read_map('COM_CMB_IQU-nilc_2048_R3.00_full.fits')#,nside)
    print('read maps ',time.time()-t1)
    cl_auto=healpy.anafast(map)
    print('got cl_auto ', time.time()-t1)
    cl_cross=healpy.anafast(m1,m2)
    print('got cl_cross ', time.time()-t1)
    cl_noise=healpy.anafast(m1-m2)
    print('got cl_noise ', time.time()-t1)
    #cl2=healpy.anafast(m1)


plt.figure(1)
plt.clf()
healpy.mollview(map*1e6,fig=1,min=-300,max=300)
plt.show()
plt.savefig('planck_nilc.png')

l=np.arange(len(cl_cross))
clx=1e12*l*(l+1)*cl_cross/2/np.pi
cln=1e12*l*(l+1)*cl_noise/2/np.pi
#cl2b=1e12*l*(l+1)*cl2/2/np.pi
ll=np.arange(len(cl_auto))
cla=1e12*ll*(ll+1)*cl_auto/2/np.pi





bl=1300
lm=2+np.arange(len(model))
bmod=model*np.exp(-0.5*lm**2/bl**2)
lm2=np.exp(-0.5*ll**2/bl**2)

lmax=3000
plt.clf()
plt.semilogy(ll,cla,'.')
plt.xlim([0,lmax])
plt.ylim([1,1e4])
plt.plot(np.arange(len(model))+2,model,'k')
plt.legend(['Raw PS','Model'])
plt.title('Raw PS')
plt.savefig('planck_ps_raw.png')

plt.clf()
plt.semilogy(ll,cla,'.')
plt.semilogy(l,clx,'.')
plt.xlim([0,lmax])
plt.ylim([1,1e4])
plt.plot(np.arange(len(model))+2,model,'k')
plt.legend(['Raw PS','X Spec','Model'])
plt.title('PS with Xcorr')
plt.savefig('planck_ps_xspec.png')

plt.clf()
plt.semilogy(ll,cla,'.')
plt.semilogy(l,clx,'.')
plt.plot(ll,cln/4,'.')
plt.xlim([0,lmax])
plt.ylim([1,1e4])
plt.plot(np.arange(len(model))+2,model,'k')
plt.legend(['Raw PS','X Spec','C Noise','Model'])
plt.title('PS with Noise')
plt.savefig('planck_ps_wnoise.png')

plt.clf()
plt.semilogy(ll,cla,'.')
plt.semilogy(l,clx,'.')
plt.plot(ll,cln/4,'.')
plt.semilogy(l,clx/lm2,'.')
plt.xlim([0,lmax])
plt.ylim([1,1e4])
plt.plot(np.arange(len(model))+2,model,'k')
plt.legend(['Raw PS','X Spec','C Noise','Beam-Divided','Model'])
plt.title('Beam-Corrected')
plt.savefig('planck_ps_wbeam.png')


