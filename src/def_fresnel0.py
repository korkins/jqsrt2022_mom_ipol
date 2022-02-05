import time
import numpy as np
from numba import jit
from def_rotator2 import rotator2
#
@jit(nopython=True, cache=True)
def fresnel0(refre, mui, mur, raz):
    '''
    Task:
        Computes 1st column of the Fresnel reflection matrix with rotation.
    In:
        refre  f   real part of refractive index
        mui    f   cos of zenith of incidence, mui > 0 (downwelling)
        mur    f   cos of zentih of reflection, mur < 0 (upwelling)
        raz    f   relative azimuth, raz = [0:2pi] rad.; forward scattering & glint @ raz = 0
    Out:
        fr1,2,3   f   IQU components
    Note:
        -
    '''
#
    fr0 = np.zeros(3)
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
#
    s2, c2 = rotator2(mui, mur, raz)
    fr0[0] = f1
    fr0[1] = f2*c2
    fr0[2] = f2*s2
#
    return fr0
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
    max_dI = 0.
    avr_dI = 0.
    max_dP = 0.
    avr_dP = 0.
#
    ix = 0
    for szai in sza:
        mui = np.cos(szai*rad) # downwelling mu > 0
        for vzai in vza:
            mur = -np.cos(vzai*rad) # upwelling mu < 0
            for razi in raz:
                fr0 = fresnel0(refre, mui, mur, razi)
                p = np.sqrt(fr0[1]*fr0[1] + fr0[2]*fr0[2])/fr0[0]
#
                bm1 = bmrk_f90[ix, 3]
                bm2 = bmrk_f90[ix, 6]
                bm3 = bmrk_f90[ix, 9]
                pbm = np.sqrt(bm2*bm2 + bm3*bm3)/bm1
#
                dI = 100.0*np.abs(fr0[0] - bm1)/bm1
                dP = 100.0*np.abs(p - pbm)
                if dI > max_dI:
                    max_dI = dI
                if dP > max_dP:
                    max_dP = dP
                avr_dI += dI
                avr_dP += dP
                ix += 1
#
    print("(in def_fresnel0): relative differences in %")
    print("(in def_fresnel0): max_dI = %.3f" %max_dI)
    print("(in def_fresnel0): avr_dI = %.3f" %(avr_dI/ix))    
    print("(in def_fresnel0): max_dP = %.3f" %max_dP)
    print("(in def_fresnel0): avr_dP = %.3f" %(avr_dP/ix))     
#
    time_end = time.time()
    print("(in def_fresnel0): runtime: %.3f sec." %(time_end-time_start))