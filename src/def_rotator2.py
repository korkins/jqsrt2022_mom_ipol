import time
import numpy as np
from numba import jit
#
@jit(nopython=True, cache=True)
def rotator2(mui, mur, raz):
    '''
    In:
        mui   f   cos of zenith of incidence; down: mui > 0, mui /= 0
        mur   f   cos of zentih of scattering (reflectance); down: mur > 0, mur /= 0
        raz   f   relative azimuth, raz = [0:2pi] rad.; forward scattering & glint @ raz = 0
    Out:
        s2x, c2x   f   sin & cos of doubled 2nd angle of rotation
    Note:
        This function is tested only for surfce reflection: mui > 0, mur < 0
    '''
#
    tiny = 1.e-8
#
    c2x = 1.0
    s2x = 0.0
    if 1.0 - mui > tiny and np.abs(np.sin(raz)) > tiny:
        caz =  np.cos(raz)
        saz = -np.sin(raz) # note '-'
        if 1.0 + mur < tiny:
            cx2 = -caz
            sx2 =  saz
        else:
            sni = np.sqrt(1.0 - mui*mui)
            sns = np.sqrt(1.0 - mur*mur)
            csca = mui*mur + sni*sns*caz
            ssca = np.sqrt(1.0 - csca*csca)
            cx2 = (mur*csca - mui)/sns/ssca
            sx2 = sni*saz/ssca
        c2x2 = cx2*cx2
        c2x = 2.0*c2x2 - 1.0
        s2x = 2.0*sx2*cx2
    return (s2x, c2x)
#==============================================================================
#
if __name__ == "__main__":
#
    time_start = time.time()
    sza = np.array([0., 30.0, 60.0, 89.0])
    vza = np.array([0., 20.0, 40.0, 60.0, 89.0])
    raz = np.array([0., 45., 90., 135., 180., 225., 270., 315., 360.])*np.pi/180.
    bmark = np.loadtxt('./benchmarks/benchmark_rotator.txt', skiprows=1) # sza, vza, raz, s1, c1, s2, c2
#
    max_diff_sin = 0.
    max_diff_cos = 0.
    avr_diff_sin = 0.
    avr_diff_cos = 0.
#
    ix = 0
    for szai in sza:
        mui = np.cos(szai*np.pi/180.0) # downwelling mu > 0
        for vzai in vza:
            mur = -np.cos(vzai*np.pi/180.0) # upwelling mu < 0
            for razi in raz:
                s2_bmrk = bmark[ix, 5]
                c2_bmrk = bmark[ix, 6]
                s2, c2 = rotator2(mui, mur, razi)
                dsin = np.abs(s2_bmrk - s2)
                dcos = np.abs(c2_bmrk - c2)
                if dsin > max_diff_sin:
                    max_diff_sin = dsin
                if dcos > max_diff_cos:
                    max_diff_cos = dcos
                avr_diff_sin += dsin
                avr_diff_cos += dcos
                ix += 1
#
    print("(in def_rotator2): benchmark format %.12f:")
    print("(in def_rotator2): max_diff_sin = %.1e" %max_diff_sin)
    print("(in def_rotator2): max_diff_cos = %.1e" %max_diff_cos)
    print("(in def_rotator2): avr_diff_sin = %.1e" %(avr_diff_sin/ix))
    print("(in def_rotator2): avr_diff_cos = %.1e" %(avr_diff_cos/ix))  
#
    time_end = time.time()
    print("(in def_rotator2): runtime: %.3f sec." %(time_end-time_start))