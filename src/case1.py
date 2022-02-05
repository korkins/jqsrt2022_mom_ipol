import time
import numpy as np
from def_add_rtls import add_rtls
from def_add_ocean import add_ocean
from def_add_lambert import add_lambert
#==============================================================================
#
if __name__ == "__main__":
    '''
    This script reads atmosphere parameters for Case 1, couples the atmopshere
    with (a) Lambertian, (b) RTLS, (c) Ocean surface and tests the result vs
    benchmarks.
    '''
#------------------------------------------------------------------------------
#
    time_start = time.time()
#
#   Files with input data and benchmark results
    path_lut = './LUTs/'
    path_bmark = './benchmarks/'
    file_Jm = path_lut + 'Jm_case1.bin'
    file_Jp = path_lut + 'Jp_case1.bin'
    file_Tm = path_lut + 'Tm_case1.bin'
    file_Rp = path_lut + 'Rp_case1.bin'
    file_metadata = path_lut + 'metadata_case1.txt'
#
#------------------------------------------------------------------------------
#   Read metadata
    with open(file_metadata, mode='r') as f:
        fcontent = f.readlines()
    Fo, tau0 = [float(value) for value in fcontent[8].split()] # line  9: Fo, tau0
    nm_srf = int(fcontent[10])                                 # line 11: nm_srf
    naz = int(fcontent[12])                                    # line 13: naz
    azd = np.zeros(naz) 
    raz = np.zeros(naz)
    for iaz in range(naz):
        line_split = fcontent[13+iaz].split()
        azd[iaz], raz[iaz] = [float(line_split[0]), float(line_split[1])]
    nmu0 = int(fcontent[19])
    sza = np.zeros(nmu0) 
    mu0 = np.zeros(nmu0)
    for imu0 in range(nmu0):
        line_split = fcontent[20+imu0].split()
        sza[imu0], mu0[imu0] = [float(line_split[0]), float(line_split[1])]
    nmu = int(fcontent[23])
    vza = np.zeros(nmu)
    mu = np.zeros(nmu)
    for imu in range(nmu):
        line_split = fcontent[24+imu].split()
        vza[imu], mu[imu] = [float(line_split[0]), float(line_split[1])]
    ng1 = int(fcontent[42])
    mug = np.zeros(ng1)
    wg = np.zeros(ng1)
    for ig1 in range(ng1):
        line_split = fcontent[43+ig1].split()
        mug[ig1], wg[ig1] = [float(line_split[0]), float(line_split[1])]
#   EOF
#------------------------------------------------------------------------------
#    
#   Read binary files
    nmu3 = nmu*3 # account for ...
    ng3 = ng1*3  # ... I Q U components
    Jm = np.fromfile(file_Jm, dtype=np.float64)   # Jm[naz x nmu0 x nmu3], lead dimension: nmu3
    Jm = np.reshape(Jm, (naz, nmu0, nmu3))        # Jm[naz, nmu0, nmu3]
    Jp = np.fromfile(file_Jp, dtype=np.float64)   # Jm[nm_srf x nmu0 x ng3]
    Jp = np.reshape(Jp, (nm_srf, nmu0, ng3))      # Jm[nm_srf, nmu0, ng3]
    Tm = np.fromfile(file_Tm, dtype=np.float64)   # Tm[nm_srf x nmu3 x ng3]
    Tm = np.reshape(Tm, (nm_srf, nmu3, ng3))      # Tm[nm_srf, nmu3, ng3]
    Rp = np.fromfile(file_Rp, dtype=np.float64)   # Rp[nm_srf x ng3 x ng3]
    Rp = np.reshape(Rp, (nm_srf, ng3, ng3))       # Tm[nm_srf, ng3, ng3]
#------------------------------------------------------------------------------
#
#   Case 1(a): add Lambertian surface and test vs becnhamrk 
    srfa = 0.3
#
    nruns = 1000
#   Lambertian surface: m = 0 only
    Jp0 = Jp[0, :, :]
    Tm0 = Tm[0, :, :]
    Rp0 = Rp[0, :, :]
    time_start_srf = time.time()
    for irun in range(nruns):
        Itoa = add_lambert(srfa, tau0, mu, mu0, raz, mug, wg, Jm, Jp0, Tm0, Rp0)
    time_end_srf = time.time()
    I = Itoa[:, :, 0:nmu3:3]
    Q = Itoa[:, :, 1:nmu3:3]
    U = Itoa[:, :, 2:nmu3:3]
    P = np.sqrt(Q*Q + U*U)/I
    dat = np.loadtxt(path_bmark+'benchmark_case1a.txt', skiprows=1)
    Ibmrk = dat[:, 3]
    Qbmrk = dat[:, 4]
    Ubmrk = dat[:, 5]
    Pbmrk = np.sqrt(Qbmrk*Qbmrk + Ubmrk*Ubmrk)/Ibmrk
    Ibmrk = np.reshape(Ibmrk, (naz, nmu0, nmu))
    Pbmrk = np.reshape(Pbmrk, (naz, nmu0, nmu))
    dI = 100.0*np.abs(I/Ibmrk - 1.0)
    dP = 100.0*np.abs(P - Pbmrk)
    print("---")
    print("Case 1(a): deviations from benchmark for Lambertian surface (in %)")
    print("-maximum: dI = %.2f,  dP = %.2f " %(np.amax(dI),  np.amax(dP)))
    print("-average: dI = %.2f,  dP = %.2f " %(np.average(dI),  np.average(dP)))
    print("runtime = %.2f sec. per %i runs" %( (time_end_srf - time_start_srf), nruns ))
#------------------------------------------------------------------------------
#
#   Case 1(b): add RTLS and test vs becnhamrk
    kl = 0.33
    kv = 0.053
    kg = 0.066
    nga = 180
#
    nruns = 100
#   Lambertian surface: m = 0 only
    time_start_srf = time.time()
    for irun in range(nruns):
        Itoa = add_rtls(nm_srf, nga, kl, kv, kg, tau0, mu, mu0, raz, mug, wg, Jm, Jp, Tm, Rp)
    time_end_srf = time.time()
    I = Itoa[:, :, 0:nmu3:3]
    Q = Itoa[:, :, 1:nmu3:3]
    U = Itoa[:, :, 2:nmu3:3]
    P = np.sqrt(Q*Q + U*U)/I
    dat = np.loadtxt(path_bmark+'benchmark_case1b.txt', skiprows=1)
    Ibmrk = dat[:, 3]
    Qbmrk = dat[:, 4]
    Ubmrk = dat[:, 5]
    Pbmrk = np.sqrt(Qbmrk*Qbmrk + Ubmrk*Ubmrk)/Ibmrk
    Ibmrk = np.reshape(Ibmrk, (naz, nmu0, nmu))
    Pbmrk = np.reshape(Pbmrk, (naz, nmu0, nmu))
    dI = 100.0*np.abs(I/Ibmrk - 1.0)
    dP = 100.0*np.abs(P - Pbmrk)
    print("---")
    print("Case 1(b): deviations from benchmark for RTLS surface (in %)")
    print("-maximum: dI = %.2f,  dP = %.2f " %(np.amax(dI),  np.amax(dP)))
    print("-average: dI = %.2f,  dP = %.2f " %(np.average(dI),  np.average(dP)))
    print("runtime = %.2f sec. per %i runs" %( (time_end_srf - time_start_srf), nruns ))
#------------------------------------------------------------------------------
#
#   Case 1(c): add Ocean and test vs becnhamrk 
    wspd = 2.
    refre = 1.33
    nga = 180
    shad = True 
#
    sgm2 = 5.12e-3*wspd + 3.0e-3
    nruns = 1
#
    time_start_srf = time.time()
    for irun in range(nruns):
        Itoa = add_ocean(nm_srf, nga, shad, sgm2, refre, tau0, mu, mu0, raz, mug, wg, Jm, Jp, Tm, Rp)
    time_end_srf = time.time()
#
    I = Itoa[:, :, 0:nmu3:3]
    Q = Itoa[:, :, 1:nmu3:3]
    U = Itoa[:, :, 2:nmu3:3]
    P = np.sqrt(Q*Q + U*U)/I
    dat = np.loadtxt(path_bmark+'benchmark_case1c.txt', skiprows=1)
    Ibmrk = dat[:, 3]
    Qbmrk = dat[:, 4]
    Ubmrk = dat[:, 5]
    Pbmrk = np.sqrt(Qbmrk*Qbmrk + Ubmrk*Ubmrk)/Ibmrk
    Ibmrk = np.reshape(Ibmrk, (naz, nmu0, nmu))
    Pbmrk = np.reshape(Pbmrk, (naz, nmu0, nmu))
    dI = 100.0*np.abs(I/Ibmrk - 1.0)
    dP = 100.0*np.abs(P - Pbmrk)
    print("---")
    print("Case 1(c): deviations from benchmark for Ocean surface (in %)")
    print("-maximum: dI = %.2f,  dP = %.2f " %(np.amax(dI),  np.amax(dP)))
    print("-average: dI = %.2f,  dP = %.2f " %(np.average(dI),  np.average(dP)))
    print("runtime = %.2f sec. per %i runs" %( (time_end_srf - time_start_srf), nruns ))
#
    time_end = time.time() 
    print("\ntotal runtime = %.2f sec." %(time_end-time_start))
#==============================================================================