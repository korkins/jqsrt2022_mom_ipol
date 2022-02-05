import time
import numpy as np
#from numba import jit
from def_gauszw import gauszw   # called if __name__ == "__main__":
from def_wavysrf import wavysrf
from def_fresnel import fresnel
#
#@jit(nopython=True, cache=True)
def oceanm(m, shad, sgm2, refre, zaz, waz, mu0, mu):
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
        mu0     f        cos of solar zenith angle, mu0 > 0 (downwelling)
        mu      f        cos of view zentih, mu < 0 (upwelling)
    Out:
        ocnm    f[3, 3]     m-th moment of the ocean reflection matrix (rowwise)
    Note:
        Below, ' if __name__ == "__main__": ' only confirms execution of
        'oceanm', but does not actually check output.
        The function was tested as dependent of add_ocean(...).
    '''
#
    ocnm = np.zeros((3, 3))
#
    nga = len(zaz)
    ocn = np.zeros((9, nga))
    for iaz, raz in enumerate(zaz):
        wsrf = wavysrf(shad, sgm2, mu0, mu, raz)
        fr = fresnel(refre, mu0, mu, raz)
        for ix in range(9):
            ocn[ix, iaz] = wsrf*fr[ix]
#
    if m > 0:
#       1st row: c c -s
        ocnm[0, 0] =  np.dot(waz*ocn[0, :], np.cos(m*zaz))
        ocnm[0, 1] =  np.dot(waz*ocn[1, :], np.cos(m*zaz))
        ocnm[0, 2] = -np.dot(waz*ocn[2, :], np.sin(m*zaz))
#       2nd row: c c -s
        ocnm[1, 0] =  np.dot(waz*ocn[3, :], np.cos(m*zaz))
        ocnm[1, 1] =  np.dot(waz*ocn[4, :], np.cos(m*zaz))
        ocnm[1, 2]  = -np.dot(waz*ocn[5, :], np.sin(m*zaz))
#       3rd row: s s  c
        ocnm[2, 0] =  np.dot(waz*ocn[6, :], np.sin(m*zaz))
        ocnm[2, 1] =  np.dot(waz*ocn[7, :], np.sin(m*zaz))
        ocnm[2, 2] =  np.dot(waz*ocn[8, :], np.cos(m*zaz))
    else:
        ocnm[0, 0] =  np.dot(waz, ocn[0, :])
        ocnm[0, 1] =  np.dot(waz, ocn[1, :])
        ocnm[1, 0] =  np.dot(waz, ocn[3, :])
        ocnm[1, 1] =  np.dot(waz, ocn[4, :])
        ocnm[2, 2] =  np.dot(waz, ocn[8, :])
#
    return ocnm/(2.0*np.pi)
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
    print("(in def_oceanm): values of 'oceanm' for different m:")
    for m in range(nm):
        ocnm = oceanm(m, shad, sgm2, refre, zaz, waz, mu0, mu)
        print("(in def_oceanm): m =%3i  %10.3e  %10.3e  %10.3e" %(m, ocnm[0, 0], ocnm[0, 1], ocnm[0, 2]))
        print("(in def_oceanm):         %10.3e  %10.3e  %10.3e" %(   ocnm[1, 0], ocnm[1, 1], ocnm[1, 2]))
        print("(in def_oceanm):         %10.3e  %10.3e  %10.3e" %(   ocnm[2, 0], ocnm[2, 1], ocnm[2, 2]))
#
    time_end = time.time()
    print("(in def_oceanm): runtime: %.3f sec." %(time_end-time_start))
#==============================================================================
# 2022/01/18:
# In:
#   shad = True
#   nga = 180
#   refre = 1.33
#   wspd = 5.
#   mu0 = 0.5
#   mu = -0.6
#   nm = 13
# Out:
# (in def_oceanm): values of 'oceanm' for different m:
# (in def_oceanm): m =  0   7.801e-02  -7.382e-02   0.000e+00
# (in def_oceanm):         -7.332e-02   7.296e-02   0.000e+00
# (in def_oceanm):          0.000e+00   0.000e+00   1.918e-02
# (in def_oceanm): m =  1   7.701e-02  -7.294e-02   3.253e-03
# (in def_oceanm):         -7.246e-02   7.215e-02  -2.392e-03
# (in def_oceanm):         -3.503e-03   2.724e-03   1.884e-02
# (in def_oceanm): m =  2   7.407e-02  -7.035e-02   6.263e-03
# (in def_oceanm):         -6.993e-02   6.975e-02  -4.617e-03
# (in def_oceanm):         -6.747e-03   5.255e-03   1.785e-02
# (in def_oceanm): m =  3   6.943e-02  -6.624e-02   8.818e-03
# (in def_oceanm):         -6.590e-02   6.594e-02  -6.526e-03
# (in def_oceanm):         -9.502e-03   7.424e-03   1.632e-02
# (in def_oceanm): m =  4   6.342e-02  -6.090e-02   1.076e-02
# (in def_oceanm):         -6.066e-02   6.095e-02  -8.007e-03
# (in def_oceanm):         -1.160e-02   9.103e-03   1.437e-02
# (in def_oceanm): m =  5   5.646e-02  -5.466e-02   1.201e-02
# (in def_oceanm):         -5.452e-02   5.509e-02  -8.995e-03
# (in def_oceanm):         -1.295e-02   1.022e-02   1.217e-02
# (in def_oceanm): m =  6   4.899e-02  -4.789e-02   1.254e-02
# (in def_oceanm):         -4.786e-02   4.868e-02  -9.473e-03
# (in def_oceanm):         -1.353e-02   1.075e-02   9.897e-03
# (in def_oceanm): m =  7   4.143e-02  -4.097e-02   1.242e-02
# (in def_oceanm):         -4.104e-02   4.205e-02  -9.473e-03
# (in def_oceanm):         -1.340e-02   1.073e-02   7.701e-03
# (in def_oceanm): m =  8   3.416e-02  -3.423e-02   1.175e-02
# (in def_oceanm):         -3.436e-02   3.552e-02  -9.063e-03
# (in def_oceanm):         -1.269e-02   1.025e-02   5.706e-03
# (in def_oceanm): m =  9   2.746e-02  -2.792e-02   1.067e-02
# (in def_oceanm):         -2.811e-02   2.933e-02  -8.337e-03
# (in def_oceanm):         -1.154e-02   9.417e-03   3.994e-03
# (in def_oceanm): m = 10   2.152e-02  -2.224e-02   9.345e-03
# (in def_oceanm):         -2.246e-02   2.367e-02  -7.399e-03
# (in def_oceanm):         -1.011e-02   8.342e-03   2.603e-03
# (in def_oceanm): m = 11   1.645e-02  -1.731e-02   7.903e-03
# (in def_oceanm):         -1.753e-02   1.868e-02  -6.349e-03
# (in def_oceanm):         -8.556e-03   7.145e-03   1.537e-03
# (in def_oceanm): m = 12   1.226e-02  -1.315e-02   6.467e-03
# (in def_oceanm):         -1.336e-02   1.441e-02  -5.279e-03
# (in def_oceanm):         -7.009e-03   5.929e-03   7.706e-04
# (in def_oceanm): runtime: 0.151 sec.
        