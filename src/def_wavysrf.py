import time
import numpy as np
#from numba import jit
from scipy import special as sp # erfc
#
#@jit(nopython=True, cache=True)
def wavysrf(shad, sgm2, mui, mur, raz):
    '''
    In:
        shad   b   if True, shadows=ON, esle - OFF
        sgm2   f   squared slope distribution paramter, 'sigma'
        mui    f   cos of zenith of incidence, mui > 0 (downwelling)
        mur    f   cos of zentih of reflection, mur < 0 (upwelling)
        raz    f   relative azimuth, raz = [0:2pi] rad.; forward scattering & glint @ raz = 0
    Out:
        roughsrf   f   scaling factor for simulation of waves
    Note:
        Different authors use different expressions for 'sigma':
            sig2 = 5.12e-3*wspd + 3.0e-3
            sig2 = ...
    '''
#
    nadir = 1.0e-8
    pi = np.pi
    sqrpi = np.sqrt(pi)
#
    mup = -mur
    caz = np.cos(raz)
    sgm = np.sqrt(sgm2)
#
    smui = np.sqrt(1.0 - mui*mui)
    if shad == False or smui < nadir:
        fi = 0.0
    else:
        x = mui/smui/sgm
        x2 = x*x
        fi = np.exp(-x2)/(sqrpi*x)
        fi = 0.5*(fi - sp.erfc(x))
#
    smur = np.sqrt(1.0 - mup*mup)
    if shad == False or smur < nadir:
        fr = 0.0
    else:
        x = mup/smur/sgm
        x2 = x*x
        fr = np.exp(-x2)/(sqrpi*x)
        fr = 0.5*(fr - sp.erfc(x))
#
    c2inc = mui*mup - smui*smur*caz
    x = mui + mup
    x2 = x*x
    a = (1.0 + c2inc)/x2
    p = a*a*np.exp((1.0 - 2.0*a)/sgm2)/(mup*sgm2)
#
    return (p/(1.0 + fi + fr))/mui
#==============================================================================
#
if __name__ == "__main__":
#
    time_start = time.time()
#
    rad = np.pi/180.
    shad = True
#
    wspd = 5
    sza = np.array([0., 45.,  60., 75.])
    vza = np.array([0., 30.,  45., 60., 80.])
    aza = np.array([0., 45., -45., 90., 135., 180., \
                    225., -225., 270., 315., 360.])
    
    bmrk_f90 = np.loadtxt('./benchmarks/benchmark_roughsrf.txt', skiprows=1) # sza, vza, aza, roughsrf
    sgm2 = 5.12e-3*wspd + 3.0e-3
    raz = aza*rad
#
    max_diff = 0.
    avr_diff = 0.
    max_wsrf = 0.
    avr_wsrf = 0.
#
    ix = 0
    for szai in sza:
        mui = np.cos(szai*rad) # downwelling mu > 0
        for vzai in vza:
            mur = -np.cos(vzai*rad) # upwelling mu < 0
            for razi in raz:
                wsrf = wavysrf(shad, sgm2, mui, mur, razi)
                diff = np.abs(bmrk_f90[ix, 3] - wsrf)
                if diff > max_diff:
                    max_diff = diff
                if wsrf > max_wsrf:
                    max_wsrf = wsrf
                avr_diff += diff
                avr_wsrf += wsrf
                ix += 1
    print("(in def_roughsrf): benchmark format %12.6e:")
    print("(in def_roughsrf): max_diff = %.1e" %max_diff)
    print("(in def_roughsrf): avr_diff = %.1e" %(avr_diff/ix))    
    print("(in def_roughsrf): max_wsrf = %.2e" %max_wsrf)
    print("(in def_roughsrf): avr_wsrf = %.2e" %(avr_wsrf/ix))     
#
    time_end = time.time()
    print("(in def_rotator2): runtime: %.3f sec." %(time_end-time_start))