import time
import numpy as np
from def_add_ocean import add_ocean
#==============================================================================
#
if __name__ == "__main__":
    '''
    This script reads atmosphere parameters for Case 2, couples the atmopshere
    with Ocean surface w/o shadowing effects and tests the result vs benchmark.
    '''
#------------------------------------------------------------------------------
#
    time_start = time.time()
#
#   Files with input data and benchmark results
    path_lut = './LUTs/'
    path_bmark = './benchmarks/'
    file_Jm = path_lut + 'Jm_case2.bin'
    file_Jp = path_lut + 'Jp_case2.bin'
    file_Tm = path_lut + 'Tm_case2.bin'
    file_Rp = path_lut + 'Rp_case2.bin'
    file_metadata = path_lut + 'metadata_case2.txt'
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
    nmu0 = int(fcontent[18])
    sza = np.zeros(nmu0) 
    mu0 = np.zeros(nmu0)
    for imu0 in range(nmu0):
        line_split = fcontent[19+imu0].split()
        sza[imu0], mu0[imu0] = [float(line_split[0]), float(line_split[1])]
    nmu = int(fcontent[22])
    vza = np.zeros(nmu)
    mu = np.zeros(nmu)
    for imu in range(nmu):
        line_split = fcontent[23+imu].split()
        vza[imu], mu[imu] = [float(line_split[0]), float(line_split[1])]
    ng1 = int(fcontent[37])
    mug = np.zeros(ng1)
    wg = np.zeros(ng1)
    for ig1 in range(ng1):
        line_split = fcontent[38+ig1].split()
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
#   Case 2: Ocean surface w/o shadowing 
    wspd = 7.
    refre = 1.34
    nga = 180
    shad = False 
#
    sgm2 = 5.12e-3*wspd + 3.0e-3
    nruns = 1
#
    time_start_srf = time.time()
    for irun in range(nruns):
        Itoa = add_ocean(nm_srf, nga, shad, sgm2, refre, tau0, mu, mu0, raz, mug, wg, Jm, Jp, Tm, Rp)
    time_end_srf = time.time()
#
#   R = pi*I/(mu0*Fo) = pi*I/mu0
    for imu0 in range(nmu0):
        Itoa[:, imu0, :] *= np.pi/mu0[imu0]
#
    I = Itoa[:, :, 0:nmu3:3]
    Q = Itoa[:, :, 1:nmu3:3]
    U = Itoa[:, :, 2:nmu3:3]
    P = np.sqrt(Q*Q + U*U)/I
    dat = np.loadtxt(path_bmark+'benchmark_case2.txt', skiprows=1)
    Ibmrk = dat[:, 3]
    Qbmrk = dat[:, 4]
    Ubmrk = dat[:, 5]
    Pbmrk = np.sqrt(Qbmrk*Qbmrk + Ubmrk*Ubmrk)/Ibmrk
    Ibmrk = np.reshape(Ibmrk, (naz, nmu0, nmu))
    Pbmrk = np.reshape(Pbmrk, (naz, nmu0, nmu))
    dI = 100.0*np.abs(I/Ibmrk - 1.0)
    dP = 100.0*np.abs(P - Pbmrk)
    print("---")
    print("Case 2: deviations from benchmark for Ocean surface (in %)")
    print("-maximum: dI = %.2f,  dP = %.2f " %(np.amax(dI),  np.amax(dP)))
    print("-average: dI = %.2f,  dP = %.2f " %(np.average(dI),  np.average(dP)))
    print("runtime = %.2f sec. per %i runs" %( (time_end_srf - time_start_srf), nruns ))
#
    time_end = time.time() 
    print("\ntotal runtime = %.2f sec." %(time_end-time_start))
#==============================================================================