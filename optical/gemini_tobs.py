import numpy as np

area=50
eff=0.5 #fraction of photons that make it through
sky_brightness=[19.61, 20.78, 21.37]
seeing=[0.6,0.85,1.1]
ab=[10,25]
t_obs=[1,100,10000]
m0=25.0 #magnitude of 1 photon per square meter per second

snr_thresh=5.0

for mag in ab:
    my_rate=10**(-0.4*(mag-m0))*eff*area
    for sky in sky_brightness:
        sky_rate_raw=10**(-0.4*(sky-m0))
        for fwhm in seeing:
            my_area=fwhm**2 #we'll go with this for the psf area
            my_sky_rate=my_area*sky_rate_raw*area*eff
            # number of good photons is my_rate*t
            #noise is sqrt((my_rate+sky_rate)*t)
            #to get to 5 sigma, takes my_rate*sqrt(t)/sqrt(my_rate+sky_rate)=snr
            #or t=(snr*sqrt(my_rate+sky_rate)/my_rate)**2
            t_obs=(snr_thresh*np.sqrt(my_rate+my_sky_rate)/my_rate)**2
            print('for sky/seeing/mag ',sky,fwhm,mag,' rates are ',my_sky_rate,my_rate,' and t_obs for ', snr_thresh,' sigma is ',t_obs)
            
#            print my_rate,my_sky_rate,mag
#            for t in t_obs:
#                nsky=my_sky_rate*t
#                nsrc=my_rate*t
 #               snr=nsrc/np.sqrt(nsrc+nsky)
  #

#fwhm2sig=np.sqrt(8*np.log(2))
