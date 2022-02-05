import time
import numpy as np
from numba import jit
#
@jit(nopython=True, cache=True)
def rtls(kl, kv, kg, mui, mur, raz):
    '''
    Task:
        To compute RTLS surface reflection for a given solar-view geometry
    In:
        kl, kv, kg   d   kernel weights
        mui, mur     d   cos of the incident and reflected zeniths; mui > 0, mur < 0 (as in RT)
        raz          d   relative azimuth in radians, as in RT: hotspot at phi = 180
    Out:
        rtls         d   brdf = kl + kv*fv + kg*fg
    Notes:
        Use raz[naz] to get naz values of rtls.
        Use one unit weight and others set to zero to generate kernel functions.
    Refs:
        1. SHARM AO 2005 Section A
    '''
#
    eps = 1.0e-12
    br = 1.0
    minc = 0.03
    hb = 2.0
    pi = np.pi
    ip = 1.0/pi
    pi2 = 0.50*pi
    pi4 = 0.25*pi
#
    ca = -np.cos(raz) # RT vs MODIS definition of zero azimuth
    ci = max(mui, minc)
    si = np.sqrt(1.0 - ci*ci)
    ti = si/ci
    cr = max(-mur, minc)
    sr = np.sqrt(1.0 - cr*cr)
    tr = sr/cr
    cx = ci*cr + si*sr*ca
    if cx > 1.0: cx = 1.0 - eps
    if cx < -1.0: cx = -1.0 + eps
    sx = np.sqrt(1.0 - cx*cx)
    x = np.arccos(cx)
#
#   Volumetric kernel
    if kv > eps:
        fv = ((pi2 - x)*cx + sx)/(ci + cr) - pi4
    else:
        fv = 0.0
#
#   Geometric kernel
    if kg > eps:
        ti = br*ti
        ti2 = ti*ti
        ci = 1.0/np.sqrt(ti2 + 1.0)
        si = np.sqrt(1.0 - ci*ci)
        tr = br*tr
        tr2 = tr*tr
        cr = 1.0/np.sqrt(tr2 + 1.0)
        sr = np.sqrt(1.0 - cr*cr)
        cx = ci*cr + si*sr*ca
        if cx > 1.0: cx = 1.0 - eps
        if cx < -1.0: cx = -1.0 + eps
#
        ic = 1.0/ci + 1.0/cr
        g2 = np.abs(ti2 + tr2 - 2.0*ti*tr*ca)
        ct = hb*np.sqrt(g2 + ti2*tr2*(1.0 - ca*ca))/ic
        if ct > 1.0: ct = 1.0 - eps
        if ct < -1.0: ct = -1.0 + eps
        t = np.arccos(ct)
        o = ip*(t - np.sqrt(1.0 - ct*ct)*ct)*ic
#
        fg = o - ic + 0.5*(1.0 + cx)/ci/cr
    else:
        fg = 0.0
#
#   Lambertian kernel
    if kl < eps:
        kl = 0.0
#
    return kl + kv*fv + kg*fg
#==============================================================================
#
if __name__ == "__main__":
#
    time_start = time.time()
#
    mui = 0.5
    mur = np.array([-1.00, -0.75, -0.50, -0.25])
#   Test 1:
    kl = 0.9
    kv = 0.5
    kg = 0.1   
    raz = 15.0*np.pi/180.0
    bmark = np.array([0.73324251549580532, 0.68990255292587532, 0.76982147875549745, 0.86359385277981926])
    print("Test 1:")
    for imur, muri in enumerate(mur):
        print(" diff ~ %.1e" %np.abs(bmark[imur] - rtls(kl, kv, kg, mui, muri, raz)))
#   Test 2:
    kl = 0.175
    kv = 0.108
    kg = 0.041
    raz = 0.0
    bmark = np.array([0.10988038334709394, 0.085458878578916805, 0.088982075844103101, 0.073684234072800836])
    print("Test 2:")
    for imur, muri in enumerate(mur):
        print(" diff ~ %.1e" %np.abs(bmark[imur] - rtls(kl, kv, kg, mui, muri, raz)))
#   Test 3:
    kl = 0.330
    kv = 0.053
    kg = 0.066
    raz = np.pi
    bmark = np.array([0.22922370664255540, 0.34517828804481682, 0.50362610015566200, 0.5529784422131629])
    print("Test 3: note 3rd value")
    for imur, muri in enumerate(mur):
        print(" diff ~ %.1e" %np.abs(bmark[imur] - rtls(kl, kv, kg, mui, muri, raz)))
#
    time_end = time.time()
    print("runtime: %.3f sec." %(time_end-time_start))
#
#==============================================================================
# 4Jan18 - Test for kernels only:
#               print rtls(1, 0.0, 1.0, 0.0, 0.3, 0.6, np.pi)
#               print rtls(1, 0.0, 0.0, 1.0, 0.3, 0.6, np.pi)
#          Tests from RTLS1.f90
#          mui = 0.5; [KL KV KG] = [0.9 0.5 0.1];
#          phi = 15*pi/180; msrf = 1
#          mur            SHARM            rtls.py
#          1.00    0.73324251549580532    0.733242515496
#          0.75    0.68990255292587532    0.689902552926
#          0.50    0.76982147875549745    0.769821478755
#          0.25    0.86359385277981926    0.86359385278
#          phi = 0, [KL KV KG] = [0.175 0.108 0.041], msrf = 1
#          1.00    0.10988038334709394    0.109880383347
#          0.75    0.085458878578916805   0.0854588785789
#          0.50    0.088982075844103101   0.0889820758441
#          0.25    0.073684234072800836   0.0736842340728
#          phi = 180, [KL KV KG] = [0.330 0.053 0.066], msrf = 1
#          1.00    0.22922370664255540    0.229223706643
#          0.75    0.34517828804481682    0.345178288045
#          0.50    0.50362610015566200    0.50362610266 <<<<<<<<< diff ~ E-9
#          0.25    0.5529784422131629     0.552978442213
#==============================================================================