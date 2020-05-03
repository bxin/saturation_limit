import numpy as np
from scipy.signal import convolve2d

def getSatSII(f, d, e, l1, l2, h, zangle, seeing, pixel_size, plate_scale):
    '''
    input parameters:
        f = focal length of telescope in meters
        d = diameter of telescope primary mirror in meters
        e = telescope obscuration ratio
        l1, l2 = apparent size of satellite in meters
        h = satellite height in meters
        zangle = zenith angle that the satellite is observed in degrees
        seeing = seeing (delivered PSF size in arcsec)
        pixel_size = pixel size in micron
        plate_scale = plate scale in arcsec per pixel
    output parameter:
        sii : satellite instantaneous image, pixel size is 1 micron
        fwhm_exp: expected FWHM using FWHM = $\sqrt{\frac{D_{sat}^2}{d^2} + \frac{D_{pupil}^2}{d^2} + \theta_{atm}^2}$, 
    '''
    fno = f/d
    cosz = np.cos(np.radians(zangle))
    range = h/cosz
    donutR=round((1/(1/f-1/range)-f)*1e6/fno/2) #in micron, trangeis is for image of a point object
    s1 = int(l1/range/np.pi*180*3600/plate_scale*pixel_size)
    s2 = int(l2/range/np.pi*180*3600/plate_scale*pixel_size)
    print('donutR = %d micron, satellite conv kernel = %d x %d microns'%(donutR, s1, s2))
    
    padding = 5 #microns
    side = int(donutR + padding) #this is actually half of side length
    x0=np.linspace(-side,side,side*2+1)
    [x, y] = np.meshgrid(x0,x0)
    r = np.sqrt(x**2+y**2)
    z = np.zeros((side*2+1, side*2+1))
    mask = np.bitwise_and((r<donutR), (r>donutR*e))
    z[mask] = 1 #image of a point object
    
    s = np.ones((s1, s2))
    f = convolve2d(z,s) #sat is a extended source
    
    sAtm = round(seeing/plate_scale*pixel_size/2.355) #sigma in micron
    side = sAtm*5 + padding  #this is actually half of side length
    x0=np.linspace(-side,side,side*2+1)
    [x, y] = np.meshgrid(x0,x0)
    r = np.sqrt(x**2+y**2)
    zAtm = np.exp(-r**2/(2*sAtm**2))
    
    sii = convolve2d(f, zAtm)
    sii = sii/np.sum(sii)
    f1 = (max(l1,l2)/range/np.pi*180*3600)**2
    f2 = (d/range/np.pi*180*3600)**2
    f3 = seeing**2
    fwhm_exp = np.sqrt(f1+f2+f3)/plate_scale*pixel_size
    print('expected FWHM = %d microns'%(fwhm_exp))
    return sii, fwhm_exp