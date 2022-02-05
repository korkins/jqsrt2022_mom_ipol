import numpy as np
from scipy import linalg as la
from def_gauszw import gauszw
from def_rtlsm import rtlsm
from def_rtls import rtls
#
def add_rtls(nm, nga, kl, kv, kg, tau0, mu, mu0, raz, mug, wg, Jm, Jp, Tm, Rp):
    '''
    Task:
        Couples atmopshere with RTLS surface
    In:
        nm         i                    number of Fourier moments
        nga        i                    order of Gaussian quadrature in Fourier expansion
        kl,kv,kg   f                    RTLS kernel weights
        tau0       f                    total atmosphere optical thickness
        mu         f[nmu]               cos of user-defined zenith angles; mu < 0
        mu0        f[nmu0]              cos of solar zenith angles; mu0 > 0
        raz        f[naz]               relative azimuths in radians
        mug        f[ng1]               nodes of Gaussian quadrature in hemisphere; mug > 0
        wg         f[ng1]               weights of Gaussian qudrature
        Jm         f[naz, nmu0, nmu3]   TOA path radiance at user-defined zeniths
        Jp         f[nm, nmu0, ng3]     Fourier moments for BOA path radiance at Gaussian nodes
        Tm         f[nm, nmu3, ng3]     Fourier moments for transmittance matrix, Tminus
        Rp         f[nm, ng3, ng3]      Fourier moments for reflectance matrix, Rplus
    Out:
        Itoa       f[naz, nmu0, nmu3]   Stokes vector at TOA
    Note:
        ng3 = 3*ng1, nmu3 = 3*nmu - to account for IQU components of the Stokes vector.
        A[nz, ny, nx] is a multidimensional array with 'nx' as a lead dimension (C-order)
    '''
#
    Itoa = np.zeros_like(Jm)
    Itoa_srf = np.zeros_like(Jm)
#
    pi = np.pi
    nmu = len(mu)
    nmu0 = len(mu0)
    naz = len(raz)
    ng1 = len(mug)
    ng3 = ng1*3
#
    rs = np.zeros((nmu, ng1))
    Rs = np.zeros((ng3, ng3))
    S0 = np.zeros(ng3)
#
    zaz, waz = gauszw(0.0, 2.0*pi, nga)
    Tdir = np.exp(tau0/mu) # direct transmittance from BOA to TOA, mu < 0
#
    scale_m12 = 1.0 # (2 - Kronecker_0m)
    for im in range(nm):
#       extract Fourier moments
        Jp_im = Jp[im, :, :]
        Rp_im = Rp[im, :, :]
        Tm_im = Tm[im, :, :]
#       matrix for single bouncing of path radiance
        for imu in range(nmu):
            for jg in range(ng1):
                rs[imu, jg] = 2.0*wg[jg]*mug[jg]* \
                                 rtlsm(im, zaz, waz, nga, kl, kv, kg, mug[jg], mu[imu]) # mu < 0
#       matrix for multiple bouncing of light
        for ig in range(ng1):
            for jg in range(ng1):
                Rs[3*ig, 3*jg] = 2.0*wg[jg]*mug[jg]* \
                                 rtlsm(im, zaz, waz, nga, kl, kv, kg, mug[jg], -mug[ig])
        I_RpRs = np.eye(ng3) - np.matmul(Rp_im, Rs)
#
        for imu0 in range(nmu0):
            cs0 = mu0[imu0]
            for ig in range(ng1):
                S0[3*ig] = cs0*np.exp(-tau0/cs0)* \
                           rtlsm(im, zaz, waz, nga, kl, kv, kg, cs0, -mug[ig])/pi
#
            Im0 = np.matmul(Tm_im, S0)                 # diffuse transmittance of S0 from BOA to TOA
            Ip = Jp_im[imu0, :] + np.matmul(Rp_im, S0) # diffuse downwelling light at BOA
            Ipb = la.solve(I_RpRs, Ip)                 # multiple bouncing of the downwelling light at BOA
            Ib = np.matmul(Rs, Ipb)                    # bouncing light is reflected ...
            Im = np.matmul(Tm_im, Ib)                  # ... and transmitted diffusely from BOA to TOA
#
            for iaz in range(naz):
                cmaz = np.cos(im*raz[iaz])*scale_m12
                smaz = np.sin(im*raz[iaz])*scale_m12
                for imu in range(nmu):
                    Itoa_srf[iaz, imu0, 3*imu  ] += (Im0[3*imu] + Im[3*imu] + \
                                                np.dot(rs[imu, :], Ipb[0:ng3:3])*Tdir[imu])*cmaz
                    Itoa_srf[iaz, imu0, 3*imu+1] += (Im0[3*imu+1] + Im[3*imu+1])*cmaz
                    Itoa_srf[iaz, imu0, 3*imu+2] += (Im0[3*imu+2] + Im[3*imu+2])*smaz # note sin(x)
#
        scale_m12 = 2.0 # (2 - Kronecker_m0); end of im-loop
#
#   accumulate all componnets on TOA ...
    Itoa = Jm + Itoa_srf
#   ... and account for bouncing of the direct soalr beam (only for I) - no Fourier expansion
    for iaz in range(naz):
        for imu0 in range(nmu0):
            cs0 = mu0[imu0]
            Tsol = np.exp(-tau0/cs0)
            for imu in range(nmu):
                Itoa[iaz, imu0, 3*imu] += cs0*Tsol*Tdir[imu]* \
                                          rtls(kl, kv, kg, cs0, mu[imu], raz[iaz])/pi
#
    return Itoa
#==============================================================================