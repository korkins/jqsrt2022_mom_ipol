import time
import numpy as np
from numba import jit
#
@jit(nopython=True, cache=True)
def rotator(mui, mur, raz):
    '''
    In:
        mui   f   cos of zenith of incidence; down: mui > 0, mui /= 0
        mur   f   cos of zentih of scattering (reflectance); down: mur > 0, mur /= 0
        raz   f   relative azimuth, raz = [0:2pi] rad.; forward scattering & glint @ raz = 0
    Out:
        s1c1s2c2   f[4]  sin & cos of 1st and 2nd angles of rotation
    Note:
        This function is tested only for surfce reflection: mui > 0, mur < 0
    '''
#
    tiny = 1.e-8
    s1c1s2c2 = np.zeros(4)
#
#   default: no rotation
    s1 = 0.0
    c1 = 1.0
    s2 = 0.0
    c2 = 1.0
    if (1.0 - mui > tiny or 1.0 + mur > tiny) and np.abs(np.sin(raz)) > tiny:
#       at leasdt one beam is far from z-axis and raz is off the principal plane
        caz =  np.cos(raz)
        saz = -np.sin(raz)
        if 1.0 - mui < tiny:   # incident beam is close to z-axis, mui > 0
            cx1 = caz
            sx1 = saz
            cx2 = 1.0
            sx2 = 0.0
        elif 1.0 + mur < tiny: # reflected beam is close to z-axis, mur < 0
            cx1 = 1.0
            sx1 = 0.0
            cx2 = -caz         # note '-'
            sx2 = saz
        else:                  # general case: both beams are far from z-axis
            sni = np.sqrt(1.0 - mui*mui)
            snr = np.sqrt(1.0 - mur*mur)
            csca = mui*mur + sni*snr*caz
            ssca = np.sqrt(1.0 - csca*csca)
            cx1 = (mui*csca - mur)/sni/ssca
            cx2 = (mur*csca - mui)/snr/ssca
            sx1 = snr*saz/ssca
            sx2 = sni*saz/ssca
        s1 = 2.0*sx1*cx1
        c1 = 2.0*cx1*cx1 - 1.0
        s2 = 2.0*sx2*cx2
        c2 = 2.0*cx2*cx2 - 1.0
    s1c1s2c2[0] = s1
    s1c1s2c2[1] = c1
    s1c1s2c2[2] = s2
    s1c1s2c2[3] = c2    
    return s1c1s2c2
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
    max_diff_sin1 = 0.
    max_diff_cos1 = 0.
    avr_diff_sin1 = 0.
    avr_diff_cos1 = 0.
    max_diff_sin2 = 0.
    max_diff_cos2 = 0.
    avr_diff_sin2 = 0.
    avr_diff_cos2 = 0.
#
    ix = 0
    for szai in sza:
        mui = np.cos(szai*np.pi/180.0) # downwelling mu > 0
        for vzai in vza:
            mur = -np.cos(vzai*np.pi/180.0) # upwelling mu < 0
            for razi in raz:
                s1_bmrk = bmark[ix, 3]
                c1_bmrk = bmark[ix, 4]
                s2_bmrk = bmark[ix, 5]
                c2_bmrk = bmark[ix, 6]
#                
                s1c1s2c2 = rotator(mui, mur, razi)
                s1 = s1c1s2c2[0]
                c1 = s1c1s2c2[1]
                s2 = s1c1s2c2[2]
                c2 = s1c1s2c2[3]
#
                dsin1 = np.abs(s1_bmrk - s1)
                dcos1 = np.abs(c1_bmrk - c1)            
                dsin2 = np.abs(s2_bmrk - s2)
                dcos2 = np.abs(c2_bmrk - c2)
                max_diff_sin1 = max(dsin1, max_diff_sin1)
                max_diff_cos1 = max(dcos1, max_diff_cos1)
                max_diff_sin2 = max(dsin2, max_diff_sin2)
                max_diff_cos2 = max(dcos2, max_diff_cos2)
 #
                avr_diff_sin1 += dsin1
                avr_diff_cos1 += dcos1
                avr_diff_sin2 += dsin2
                avr_diff_cos2 += dcos2
                ix += 1
#
    print("(in def_rotator): benchmark format %.12f:")
    print("(in def_rotator): max_diff_sin1 = %.1e" %max_diff_sin1)
    print("(in def_rotator): max_diff_cos1 = %.1e" %max_diff_cos1)
    print("(in def_rotator): max_diff_sin2 = %.1e" %max_diff_sin2)
    print("(in def_rotator): max_diff_cos2 = %.1e" %max_diff_cos2)
    print("(in def_rotator): avr_diff_sin1 = %.1e" %(avr_diff_sin1/ix))
    print("(in def_rotator): avr_diff_cos1 = %.1e" %(avr_diff_cos1/ix))
    print("(in def_rotator): avr_diff_sin2 = %.1e" %(avr_diff_sin2/ix))
    print("(in def_rotator): avr_diff_cos2 = %.1e" %(avr_diff_cos2/ix))
#
    time_end = time.time()
    print("(in def_rotator2): runtime: %.3f sec." %(time_end-time_start))