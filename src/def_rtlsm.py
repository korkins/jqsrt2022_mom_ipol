import time
import numpy as np
from numba import jit
from def_gauszw import gauszw # if __name__ == "__main__":
from def_rtls import rtls
#
@jit(nopython=True, cache=True)
def rtlsm(m, zaz, waz, nga, kl, kv, kg, mui, mur):
    '''
    Task:
        To compute m-th Fourier moment for RTLS model for the given pair of directions
        of incidence and reflection.
    In:
        m      i        Fourier moment, m = 0, 1, 2... (as in RT)
	    zaz    d[nga]   Gauss nodes in (0:2pi) - relative azimuth in radians
	    waz    d[nga]   Gauss weights
	    nga    i        Order of Gauss qudrature for integration over azimuth, nga > 1
	    kl     d        Lambertian kernel weight, kl = [0, 1]
	    kv     d        volumetric kernel weight, kv = [-1, 1] (?)
	    kg     d        geometric-optics kernel weight, kg = [-1, 1] (?)
	    mui    d        cos of incidence zenith, cmui > 0.0 (down) - as in RTE
	    mur    d        cos of reflection zenith, cmur < 0.0 (up) - as in RTE
    Out:
        srfm   d   Fourier moment, (2-d0m) is *not* included
    Notes:
        Kronecker delta, (2-d0m), is *not* included
    Refs:
        1. https://en.wikipedia.org/wiki/Fourier_series
    '''
#
    srf = np.zeros(nga)
    for iaz, raz in enumerate(zaz):
        srf[iaz] = rtls(kl, kv, kg, mui, mur, raz)
#
    if m > 0:
        srfm = np.dot(waz*srf, np.cos(m*zaz))
    else:
        srfm = np.dot(waz, srf)
#
    return srfm/(2.0*np.pi)
#==============================================================================
#
if __name__ == "__main__":
#
    time_start = time.time()
#
    mui = 0.5
    mur = -0.5
    kl = 0.330
    kv = 0.053
    kg = 0.066
    nga = 180
    nm = 50
    aza = np.array([0.0, 45.0, 90.0, 135.0, 180.0])
#
    zaz, waz = gauszw(0.0, 2.0*np.pi, nga)
#
    for raz in aza*np.pi/180:
#       exact surface
        srf0 = rtls(kl, kv, kg, mui, mur, raz)
#       Fourier summation
        srf = rtlsm(0, zaz, waz, nga, kl, kv, kg, mui, mur)  
        for im in range(1, nm):
            srf += 2.0*rtlsm(im, zaz, waz, nga, kl, kv, kg, mui, mur)*np.cos(im*raz)
        print("raz = %5.1f,  diff = %5.2f o/o" %( raz*180.0/np.pi, 100.0*(srf/srf0 - 1.0) ))
#
    time_end = time.time()
    print("runtime: %.3f sec." %(time_end-time_start))
#
#==============================================================================