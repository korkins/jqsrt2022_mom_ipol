import time
import numpy as np
#from numba import jit
from def_gauszw import gauszw # called if __name__ == "__main__":
from def_wavysrf import wavysrf
from def_fresnel0 import fresnel0
#
#@jit(nopython=True, cache=True)
def oceanm0(m, shad, sgm2, refre, zaz, waz, mu0, mu):
    '''
    Task:
        Computes m-th Fourier moment for the 1st column of the ocean reflection
        Mueller matrix.
    In:
        m       i        Fourier moment, m = 0, 1, 2 ...
        shad    b        if True, shadows = ON, else - OFF
        sgm2    f        squared slope distribution paramter, 'sigma'
        refre   f        real part of refractive index
        zaz     d[nga]   Gauss nodes in (0:2pi) - relative azimuth in radians
	    waz     d[nga]   Gauss weights
        wspd    f        wind speed, m/s
        mu0     f        cos of solar zenith angle, mu0 > 0 (downwelling)
        mu      f        cos of view zentih, mu < 0 (upwelling)
    Out:
        ocnm0   f[3]     1st column of the matrix 
    Note:
        Different authors use different expressions for 'sigma':
            sig2 = 5.12e-3*wspd + 3.0e-3
            sig2 = ...
        Below, ' if __name__ == "__main__": ' only confirms execution of
        'oceanm0', but does not actually check output.
    '''
#
    ocnm0 = np.zeros(3)
#
    nga = len(zaz)
    ocn = np.zeros((3, nga))
    for iaz, raz in enumerate(zaz):
        wsrf = wavysrf(shad, sgm2, mu0, mu, raz)
        fr0 = fresnel0(refre, mu0, mu, raz)
        ocn[0, iaz] = wsrf*fr0[0]
        ocn[1, iaz] = wsrf*fr0[1]
        ocn[2, iaz] = wsrf*fr0[2]
#
    if m > 0:
        ocnm0[0] = np.dot(waz*ocn[0, :], np.cos(m*zaz))
        ocnm0[1] = np.dot(waz*ocn[1, :], np.cos(m*zaz))
        ocnm0[2] = np.dot(waz*ocn[2, :], np.sin(m*zaz))
    else:
        ocnm0[0] = np.dot(waz, ocn[0, :])
        ocnm0[1] = np.dot(waz, ocn[1, :])
#
    return ocnm0/(2.0*np.pi)
#==============================================================================
#
if __name__ == "__main__":
#
    time_start = time.time()
#
    shad = True
    nga = 180
    refre = 1.33
    wspd = 5.
    mu0 = 0.5
    mu = -0.6
    nm = 13
#
    zaz, waz = gauszw(0.0, 2.0*np.pi, nga)
#
    sgm2 = 5.12e-3*wspd + 3.0e-3
    print("(in def_oceanm0): values of 'oceanm0' for different m:")
    for m in range(nm):
        ocnm0 = oceanm0(m, shad, sgm2, refre, zaz, waz, mu0, mu)
        print("(in def_oceanm0): m =%3i  %10.3e  %10.3e  %10.3e" %(m, ocnm0[0], ocnm0[1], ocnm0[2]))
#
    time_end = time.time()
    print("(in def_fresnel): runtime: %.3f sec." %(time_end-time_start))
#==============================================================================
# 2022/18/01:
# In:
#    shad = True
#    nga = 180
#    refre = 1.33
#    wspd = 5.
#    mu0 = 0.5
#    mu = -0.6
#    nm = 1
# Out:
# (in def_oceanm0): values of 'oceanm0' for different m:
# (in def_oceanm0): m =  0   7.801e-02  -7.332e-02   0.000e+00
# (in def_oceanm0): m =  1   7.701e-02  -7.246e-02  -3.503e-03
# (in def_oceanm0): m =  2   7.407e-02  -6.993e-02  -6.747e-03
# (in def_oceanm0): m =  3   6.943e-02  -6.590e-02  -9.502e-03
# (in def_oceanm0): m =  4   6.342e-02  -6.066e-02  -1.160e-02
# (in def_oceanm0): m =  5   5.646e-02  -5.452e-02  -1.295e-02
# (in def_oceanm0): m =  6   4.899e-02  -4.786e-02  -1.353e-02
# (in def_oceanm0): m =  7   4.143e-02  -4.104e-02  -1.340e-02
# (in def_oceanm0): m =  8   3.416e-02  -3.436e-02  -1.269e-02
# (in def_oceanm0): m =  9   2.746e-02  -2.811e-02  -1.154e-02
# (in def_oceanm0): m = 10   2.152e-02  -2.246e-02  -1.011e-02
# (in def_oceanm0): m = 11   1.645e-02  -1.753e-02  -8.556e-03
# (in def_oceanm0): m = 12   1.226e-02  -1.336e-02  -7.009e-03
# (in def_fresnel): runtime: 0.086 sec.