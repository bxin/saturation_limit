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
        seeing = seeing (delivered PSF size in arcsec) at the given zangle
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
    print('padding = ', padding)
    side = int(donutR + padding) #this is actually half of side length
    x0=np.linspace(-side,side,side*2+1)
    [x, y] = np.meshgrid(x0,x0)
    r = np.sqrt(x**2+y**2)
    z = np.zeros((side*2+1, side*2+1))
    mask = np.bitwise_and((r<=donutR), (r>=donutR*e))
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

def getSatSIIvonK(f, d, e, l1, l2, h, zangle, seeing, pixel_size, plate_scale):
    '''
    input parameters:
        f = focal length of telescope in meters
        d = diameter of telescope primary mirror in meters
        e = telescope obscuration ratio
        l1, l2 = apparent size of satellite in meters
        h = satellite height in meters
        zangle = zenith angle that the satellite is observed in degrees
        seeing = seeing (delivered PSF size in arcsec) at the given zangle
        pixel_size = pixel size in micron
        plate_scale = plate scale in arcsec per pixel
    output parameter:
        sii : satellite instantaneous image, 
        array_pixel_size: pixel size for sii, in microns. For LSST, this is 3.5 micron (=0.07 arcsec = 0.35 LSST pixel)
        fwhm_exp: expected FWHM using FWHM = $\sqrt{\frac{D_{sat}^2}{d^2} + \frac{D_{pupil}^2}{d^2} + \theta_{atm}^2}$, 
    '''
    #zAtm = np.loadtxt('vonK1.0.txt') #this was generated for 1.0" FWHM, pixel size=0.1". We squeeze it 0.7" FWHM, pixel size=0.07".
    zAtm = np.loadtxt('vonK1.0_2k.txt') #this was generated for 1.0" FWHM, pixel size=0.1". We squeeze it 0.7" FWHM, pixel size=0.07".
    array_pixel_size = 0.1*seeing/plate_scale*pixel_size #0.1" per pixel was for the vonK file loaded above
    
    fno = f/d
    cosz = np.cos(np.radians(zangle))
    range = h/cosz
    donutR=round((1/(1/f-1/range)-f)*1e6/fno/2/array_pixel_size) #in 3.5 micron pixel
    s1 = int(l1/range/np.pi*180*3600/plate_scale*pixel_size/array_pixel_size)
    s2 = int(l2/range/np.pi*180*3600/plate_scale*pixel_size/array_pixel_size)
    #print('donutR = %d pixel (1pix = 0.07arcsec), satellite conv kernel = %d x %d microns'%(donutR, s1, s2))
    
    padding = np.round((zAtm.shape[0] - s1 - donutR*2)/2) #480 #pixels
    #print('padding = ', padding)
    side = int(donutR + padding) #this is actually half of side length
    x0=np.linspace(-side,side,side*2+1)
    [x, y] = np.meshgrid(x0,x0)
    r = np.sqrt(x**2+y**2)
    z = np.zeros((side*2+1, side*2+1))
    mask = np.bitwise_and((r<=donutR), (r>=donutR*e))
    z[mask] = 1 #image of a point object
    
    s = np.ones((s1, s2))
    f = convolve2d(z,s) #sat is a extended source
    #print(z.shape, s.shape, f.shape)
    
    #sii = convolve2d(f, zAtm)
    ffft = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(f), s=zAtm.shape))
    zfft = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(zAtm), s=zAtm.shape))
    prodfft = ffft*zfft
    sii = np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(prodfft),
                                                    s=prodfft.shape))
    
    sii = np.absolute(sii)
    sii = sii/np.sum(sii)
    f1 = (max(l1,l2)/range/np.pi*180*3600)**2
    f2 = (d/range/np.pi*180*3600)**2
    f3 = seeing**2
    fwhm_exp = np.sqrt(f1+f2+f3)/plate_scale*pixel_size
    #print('expected FWHM = %d microns'%(fwhm_exp))#this is still in micron
    return sii, fwhm_exp, array_pixel_size

def findWidth(cs, y):
    '''
    input:
        cs is the input 1D profile
        y is the threshold
    output:
        the width as defined by the threshold y, in the unit of x-pixels
    '''
    peak = max(cs)
    idx_peak1 = np.argmax(abs(cs-peak)/peak<0.001)
    idx_peak2 = len(cs) - np.argmax(abs(cs[::-1]-peak)/peak<0.001)-1
    #print(idx_peak1, len(cs), idx_peak2)
    valley_width = idx_peak2-idx_peak1
    x1=np.arange(idx_peak1+1)
    curve1 = cs[x1]
    if valley_width > 0:
        valley = min(cs[idx_peak1:idx_peak2])
        idx_valley = np.argmax(cs==valley)
        x2=np.arange(idx_peak1, idx_valley)
        curve2 = cs[x2][::-1]
        x2 = x2-idx_peak1
    if y>peak:
        width = 0
    else:
        x = np.interp(y, curve1, x1)
        if valley_width > 0:
            width = idx_valley - x
            # comment out 2 lines below, if pixels inside the valley will not be used.
            if y>valley:
                width = width - np.interp(y, curve2, x2)
        else:
            width = idx_peak1 - x
        width = width*2
    return width