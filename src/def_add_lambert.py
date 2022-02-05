import numpy as np
from scipy import linalg as la
#
def add_lambert(srfa, tau0, mu, mu0, raz, mug, wg, Jm, Jp, Tm, Rp):
    '''
    Task:
        To attach isotropic surface to atmopshere from IPOL.
    In:
        srfa   f         Lambertian surface albedo
        tau0   f         total atmosphere optical thickness
        mu     f[nmu]    cos of user-defined zenith angles; mu < 0
        mu0    f[nmu0]   cos of solar zenith angles; mu0 > 0
        raz    f[naz]    relative azimuths in radians
        mug    f[ng1]    nodes of Gaussian quadrature in hemisphere; mug > 0
        wg     f[ng1]    weights of Gaussian qudrature
        Jm     f[naz, nmu0, nmu3]   TOA path radiance at user-defined zeniths
        Jp     f[nmu0, ng3]   m=0 Fourier moment for BOA path radiance at Gaussian nodes
        Tm     f[nmu3, ng3]   m=0 Fourier moment for transmittance matrix, Tminus
        Rp     f[ng3, ng3]    m=0 Fourier moment for reflectance matrix, Rplus
    Out:
        Itoa   f[naz, nmu0, nmu3]   Stokes vector at TOA
    Note:
        ng3 = 3*ng1, nmu3 = 3*nmu - to account for the Stokes-IQU.
        A[nz, ny, nx] is a multidimensional array with 'nx' as a lead dimension (C-order)
    '''
#
    Itoa = np.zeros_like(Jm)
#
    pi = np.pi
    nmu = len(mu)
    nmu0 = len(mu0)
    naz = len(raz)
    ng1 = len(mug)
    ng3 = ng1*3
#
    rs = np.zeros(ng3)
    rs[0:ng3:3] = 2.0*srfa*wg*mug
    Rs = np.zeros((ng3, ng3))
    for ig in range(ng1):
        Rs[3*ig, :] = rs
    I_RpRs = np.eye(ng3) - np.matmul(Rp, Rs)
#
    S0 = np.zeros(ng3)
    for imu0 in range(nmu0):
        cs0 = mu0[imu0]
        for ig in range(ng1):
            S0[3*ig] = cs0*srfa*np.exp(-tau0/cs0)/pi # bouncing of direct solar beam
#
        Im0 = np.matmul(Tm, S0)              # diffuse transmittance of S0 from BOA to TOA
        Ip = Jp[imu0, :] + np.matmul(Rp, S0) # diffuse downwelling light at BOA
        Ipb = la.solve(I_RpRs, Ip)           # multiple bouncing of the downwelling light at BOA
        Ib = np.matmul(Rs, Ipb)              # bouncing light is reflected ...
        Im = np.matmul(Tm, Ib)               # ... and transmitted diffusely from BOA to TOA
#
        for iaz in range(naz):
            for imu in range(nmu):
                Tdir = np.exp(tau0/mu[imu])  # direct transmittance from BOA to TOA, mu < 0
                Itoa[iaz, imu0, 3*imu  ] = Jm[iaz, imu0, 3*imu] + Im0[3*imu] + Im[3*imu] + \
                                           2.0*srfa*np.dot(mug*wg, Ipb[0:ng3:3])*Tdir + \
                                           cs0*srfa*np.exp(-tau0/cs0)*Tdir/pi
#               Lambertian reflection does not contribute to Q and U
                Itoa[iaz, imu0, 3*imu+1] = Jm[iaz, imu0, 3*imu+1]
                Itoa[iaz, imu0, 3*imu+2] = Jm[iaz, imu0, 3*imu+2]
#   
    return Itoa
#==============================================================================