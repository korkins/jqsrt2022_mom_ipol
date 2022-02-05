import time
import numpy as np
from numba import jit
from def_rotator import rotator
#
@jit(nopython=True, cache=True)
def fresnel(refre, mui, mur, raz):
    '''
    In:
        refre  f   real part of refractive index
        mui    f   cos of zenith of incidence, mui > 0 (downwelling)
        mur    f   cos of zentih of reflection, mur < 0 (upwelling)
        raz    f   relative azimuth, raz = [0:2pi] rad.; forward scattering & glint @ raz = 0
    Out:
        fr     f[9]   3 x 3 Fresnel matrix with rotation
    Note:
        fr is stored rowwise (C-order): fr11 fr12 fr13 ... fr32 fr33
    '''
#
    fr = np.zeros(9)
#
    caz = np.cos(raz)
    mui2 = mui*mui
    mur2 = mur*mur
    si = np.sqrt(1.0 - mui2)
    sr = np.sqrt(1.0 - mur2)
#
    cinc2 = -mui*mur - si*sr*caz
#
    c2inc = 0.5 + 0.5*cinc2
    cinc = np.sqrt(c2inc)
#
    nri2 = refre*refre
    ca = np.sqrt(nri2 - 1.0 + c2inc)
    cb = nri2*cinc
    rl = (cb - ca)/(cb + ca)
    rr = (cinc - ca)/(cinc + ca)
#
    rl2 = rl*rl
    rr2 = rr*rr
    f1 = 0.5*(rl2 + rr2)
    f2 = 0.5*(rl2 - rr2)
    f3 = rl*rr
#
    s1c1s2c2 = rotator(mui, mur, raz)
    s1 = s1c1s2c2[0]
    c1 = s1c1s2c2[1]
    s2 = s1c1s2c2[2]
    c2 = s1c1s2c2[3]
#
#   1st row
    fr[0] =  f1
    fr[1] =  f2*c1
    fr[2] = -f2*s1
#   2nd row
    fr[3] =  f2*c2
    fr[4] =  f1*c1*c2 - f3*s1*s2
    fr[5] = -f1*s1*c2 - f3*c1*s2
#   3rd row
    fr[6] =  f2*s2
    fr[7] =  f1*c1*s2 + f3*s1*c2
    fr[8] = -f1*s1*s2 + f3*c1*c2
#
    return fr
#==============================================================================
#
if __name__ == "__main__":
#
    time_start = time.time()
#
    rad = np.pi/180.
#
    refre = 1.33
    sza = np.array([0., 45.,  60., 75.])
    vza = np.array([0., 30.,  45., 60., 80.])
    aza = np.array([0., 45., -45., 90., 135., 180., \
                    225., -225., 270., 315., 360.])
#
#   sza, vza, aza, f11, f12, f13, f21, f22, f23, f31, f32, f33
    bmrk_f90 = np.loadtxt('./benchmarks/benchmark_fresnel.txt', skiprows=1)
    raz = aza*rad
#
    maxij = np.zeros(9)
    avrij = np.zeros(9)
    difij = np.zeros(9)
    ix = 0
    for szai in sza:
        mui = np.cos(szai*rad) # downwelling mu > 0
        for vzai in vza:
            mur = -np.cos(vzai*rad) # upwelling mu < 0
            for razi in raz:
                fr = fresnel(refre, mui, mur, razi)
                fr[1:] /= fr[0]
#
                bm = bmrk_f90[ix, 3:]
                bm[1:] /= bm[0]
 #              
                difij[0 ] = 100.0*np.abs(fr[0] - bm[0])/bm[0]
                difij[1:] = 100.0*np.abs(fr[1:] - bm[1:])
#
                avrij += difij
                maxij = np.maximum(maxij, difij)
                ix += 1
#
    print("(in def_fresnel): relative difference in %, benchmark fmt = %.6e")
    print("(in def_fresnel): max, 1st row: %.1e  %.1e  %.1e" %(maxij[0], maxij[1], maxij[2]))
    print("(in def_fresnel): max, 2nd row: %.1e  %.1e  %.1e" %(maxij[3], maxij[4], maxij[5]))
    print("(in def_fresnel): max, 3rd row: %.1e  %.1e  %.1e" %(maxij[6], maxij[7], maxij[8]))
#
    avrij /= ix
    print("(in def_fresnel): avr, 1st row: %.1e  %.1e  %.1e" %(avrij[0], avrij[1], avrij[2]))
    print("(in def_fresnel): avr, 2nd row: %.1e  %.1e  %.1e" %(avrij[3], avrij[4], avrij[5]))   
    print("(in def_fresnel): avr, 3rd row: %.1e  %.1e  %.1e" %(avrij[6], avrij[7], avrij[8]))
#
    time_end = time.time()
    print("(in def_fresnel): runtime: %.3f sec." %(time_end-time_start))