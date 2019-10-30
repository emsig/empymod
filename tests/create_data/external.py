r"""Routines to create data from external modellers, for comparison purposes.

- DIPOLE1D: You must have Dipole1D installed and it must be in your system
  path; https://software.seg.org/2012/0003.

- EMmod: You must have Dipole1D installed and it must be in your system
  path; https://software.seg.org/2015/0001.

- Green3D: You must have Green3D installed (for which you need to be a member
  of the CEMI consortium). The following files must be in the folder
  `empymod/tests/green3d`: `green3d.m`, `grint.mexa64`,
  `grint.mexw64`,`normal.mexa64`, and `normal.mexw64`. Furthermore, you need
  Matlab.

Tested only on Linux (Ubuntu 16.04 LTS, x86_64).

Warning: These functions are to generate test-data with the provided scripts.
         They do not check the input, and are therefore very fragile if you do
         not provide the input as expected.

"""
import os
import subprocess
import numpy as np
from scipy.constants import mu_0
from os.path import join, dirname


class ChDir(object):
    r"""Step into a directory temporarily.

    Taken from:
    pythonadventures.wordpress.com/2013/12/15/
                    chdir-a-context-manager-for-switching-working-directories

    """

    def __init__(self, path):
        self.old_dir = os.getcwd()
        self.new_dir = path

    def __enter__(self):
        os.chdir(self.new_dir)

    def __exit__(self, *args):
        os.chdir(self.old_dir)


def green3d(src, rec, depth, res, freq, aniso, par, strength=0):
    r"""Run model with green3d (CEMI).

    You must have Green3D installed (for which you need to be a member of the
    CEMI consortium). The following files must be in the folder
    `empymod/tests/green3d`:
        - `green3d.m`
        - `grint.mexa64`
        - (`grint.mexw64`)
        - (`normal.mexa64`)
        - (`normal.mexw64`).
    Furthermore, you need to have Matlab installed.

    http://www.cemi.utah.edu

    """

    # Execution directory
    # (no need to create it, it HAS to exist with the necessary green3d-files).
    rundir = join(dirname(__file__), 'tmp/green3d/')

    # Source-input depending on par
    if par in [9, 10]:
        srcstr = str(src[0]) + ' ' + str(src[1]) + ' ' + str(src[2]) + ' '
        srcstr += str(src[3]) + ' ' + str(src[4])
    elif par in [2, 3]:
        srcstr = str(strength) + ' ' + str(src[0]) + ' ' + str(src[2]) + ' '
        srcstr += str(src[4]) + ' ' + str(src[1]) + ' ' + str(src[3]) + ' '
        srcstr += str(src[5])
    elif par in [6, 7, 8]:
        srcstr = str(src[0]) + ' ' + str(src[1]) + ' ' + str(src[2])

    # Write input file
    with open(rundir + 'run.sh', 'wb') as runfile:

        runfile.write(bytes(
            '#!/bin/bash\n\nmatlab -nodesktop -nosplash -r "[e, h] = green3d('
            '[' + ','.join(map(str, freq))+'], '
            '[' + ','.join(map(str, depth[1:] - np.r_[0, depth[1:-1]])) + '], '
            '[' + ','.join(map(str, 1/res[1:])) + '], '
            '[' + ','.join(map(str, aniso[1:])) + '], '
            '[' + ','.join(map(str, rec[0].ravel())) + '], '
            '[' + ','.join(map(str, rec[1].ravel())) + '], '
            '[' + ','.join(map(str, np.ones(np.size(rec[0])) * rec[2])) + '], '
            '[' + str(par) + ' ' + srcstr + ']); exit"', 'UTF-8'))

    # Run Green3D
    with ChDir(rundir):
        subprocess.run('bash run.sh', shell=True,
                       stderr=subprocess.STDOUT, stdout=subprocess.PIPE)

    # Read output-file
    with open(rundir + 'out.txt', 'rb') as outfile:
        temp = np.loadtxt(outfile)

        Ex = temp[:, 0] + 1j*temp[:, 1]
        Ey = temp[:, 2] + 1j*temp[:, 3]
        Ez = temp[:, 4] + 1j*temp[:, 5]
        Hx = temp[:, 6] + 1j*temp[:, 7]
        Hy = temp[:, 8] + 1j*temp[:, 9]
        Hz = temp[:, 10] + 1j*temp[:, 11]

        if par in [6, 7, 8, 10]:
            Ex /= 2j*freq*np.pi*mu_0
            Ey /= 2j*freq*np.pi*mu_0
            Ez /= 2j*freq*np.pi*mu_0
            Hx /= 2j*freq*np.pi*mu_0
            Hy /= 2j*freq*np.pi*mu_0
            Hz /= 2j*freq*np.pi*mu_0

        return Ex, Ey, Ez, Hx, Hy, Hz


def dipole1d(src, rec, depth, res, freq, srcpts=5):
    r"""Run model with dipole1d (Scripps).

    You must have Dipole1D installed and it must be in your system path.

    https://software.seg.org/2012/0003

    """

    # Create directory, overwrite existing
    rundir = join(dirname(__file__), 'tmp/dipole1d/')
    os.makedirs(rundir, exist_ok=True)

    # Source: A bipole in dipole1d is defined as: center point, angles, length
    if len(src) == 6:
        dx = src[1] - src[0]
        dy = src[3] - src[2]
        dz = src[5] - src[4]
        r = np.sqrt(dx**2 + dy**2 + dz**2)
        theta = np.rad2deg(np.arctan2(dy, dx))
        phi = np.rad2deg(np.pi/2-np.arccos(dz/r))
        src = [src[0]+dx/2, src[2]+dy/2, src[4]+dz/2, theta, phi]

    else:
        r = 0  # 0 = dipole

    # Angle: In empymod, x is Easting, and the angle is the deviation from x
    #        anticlockwise.  In Dipole1D, x is Northing, and the angle is the
    #        deviation from x clockwise. Convert angle to within 0<=ang<360:
    ang = (-src[3] % - 360 + 90) % 360

    # Counts
    nsrc = np.size(src[2])
    nrec = np.size(rec[0])
    nfreq = np.size(freq)
    nlay = np.size(res)

    # Write input file
    with open(rundir + 'RUNFILE', 'wb') as runfile:
        runfile.write(bytes(
            'Version:          DIPOLE1D_1.0\n'
            'Output Filename:  dipole1d.csem\n'
            'CompDerivatives:  no\n'
            'HT Filters:       kk_ht_401\n'
            'UseSpline1D:      no\n'
            'Dipole Length:    '+str(r)+'\n'
            '# integ pts:      '+str(srcpts)+'\n'
            '# TRANSMITTERS:   '+str(nsrc)+'\n'
            '          X           Y           Z    ROTATION         DIP\n',
            'UTF-8'))
        np.savetxt(runfile, np.atleast_2d(np.r_[src[1], src[0], src[2], ang,
                   src[4]]), fmt='%12.4f')
        runfile.write(bytes('# FREQUENCIES:    '+str(nfreq)+'\n',
                      'UTF-8'))
        np.savetxt(runfile, freq, fmt='%10.3f')
        runfile.write(bytes('# LAYERS:         '+str(nlay)+'\n',
                      'UTF-8'))
        np.savetxt(runfile, np.r_[[np.r_[-1000000, depth]], [res]].transpose(),
                   fmt='%12.5g')
        runfile.write(bytes('# RECEIVERS:      '+str(nrec)+'\n',
                      'UTF-8'))
        rec = np.r_[[rec[1].ravel()], [rec[0].ravel()],
                    [np.ones(np.size(rec[0]))*rec[2]]]
        np.savetxt(runfile, rec.transpose(), fmt='%12.4f')

    # Run dipole1d
    with ChDir(rundir):
        subprocess.run('DIPOLE1D', shell=True,
                       stderr=subprocess.STDOUT, stdout=subprocess.PIPE)

    # Read output-file
    skiprows = nlay + nsrc + nfreq + nrec + 6
    with open(rundir + 'dipole1d.csem', 'rb') as outfile:
        temp = np.loadtxt(outfile, skiprows=skiprows, unpack=True)
        Ex = temp[0] - 1j*temp[1]
        Ey = temp[2] - 1j*temp[3]
        Ez = temp[4] - 1j*temp[5]
        Hx = -temp[6]/mu_0 + 1j*temp[7]/mu_0
        Hy = -temp[8]/mu_0 + 1j*temp[9]/mu_0
        Hz = -temp[10]/mu_0 + 1j*temp[11]/mu_0

    return Ey, Ex, Ez, Hy, Hx, Hz


def emmod(dx, nx, dy, ny, src, rec, depth, res, freq, aniso, epermV, epermH,
          mpermV, mpermH, ab, nd=1000, startlogx=-6, deltalogx=0.5, nlogx=24,
          kmax=10, c1=0, c2=0.001, maxpt=1000, dopchip=0, xdirect=0):
    r"""Run model with emmod (Hunziker et al, 2015).

    You must have EMmod installed and it must be in your system path.

    https://software.seg.org/2015/0001

    nd        : number of integration domains
    startlogx : first integration point in space
    deltalogx : log sampling rate of integr. pts in space at first iteration
    nlogx     : amount of integration points in space at first iteration
    kmax      : largest wavenumber to be integrated
    c1        : first precision parameter
    c2        : second precision parameter
    maxpt     : maximum amount of integration points in space
    dopchip   : pchip interpolation (1) or linear interpolation (0)
    xdirect   : direct field in space domain (1) or in wavenumber domain (0)

    """

    # Create directory, overwrite existing
    rundir = join(dirname(__file__), 'tmp/emmod/')
    os.makedirs(rundir, exist_ok=True)

    # Write input-file
    with open(rundir + 'emmod.scr', 'wb') as runfile:
        runfile.write(bytes(
            '#!/bin/bash\n\nemmod \\\n'
            '  freq='+str(freq)+' \\\n'
            '  file_out=emmod.out \\\n'
            '  writebin=0 \\\n'
            '  nx='+str(nx)+' \\\n'
            '  ny='+str(ny)+' \\\n'
            '  zsrc='+str(src[2])+' \\\n'
            '  zrcv='+str(rec[2])+' \\\n'
            '  dx='+str(dx)+' \\\n'
            '  dy='+str(dy)+' \\\n'
            '  z='+','.join(map(str, np.r_[-1, depth]))+' \\\n'
            '  econdV='+','.join(map(str, 1/(res*aniso**2)))+' \\\n'
            '  econdH='+','.join(map(str, 1/res))+' \\\n'
            '  epermV='+','.join(map(str, epermV))+' \\\n'
            '  epermH='+','.join(map(str, epermH))+' \\\n'
            '  mpermV='+','.join(map(str, mpermV))+' \\\n'
            '  mpermH='+','.join(map(str, mpermH))+' \\\n'
            '  verbose=0 \\\n'
            '  component='+str(ab)+' \\\n'
            '  nd='+str(nd)+' \\\n'
            '  startlogx='+str(startlogx)+' \\\n'
            '  deltalogx='+str(deltalogx)+' \\\n'
            '  nlogx='+str(nlogx)+' \\\n'
            '  kmax='+str(kmax)+' \\\n'
            '  c1='+str(c1)+' \\\n'
            '  c2='+str(c2)+' \\\n'
            '  maxpt='+str(maxpt)+' \\\n'
            '  dopchip='+str(dopchip)+' \\\n'
            '  xdirect='+str(xdirect)+' \n',
            'UTF-8'))

    # Run EMmod
    with ChDir(rundir):
        subprocess.run('bash emmod.scr', shell=True,
                       stderr=subprocess.STDOUT, stdout=subprocess.PIPE)

    # Read output-file
    with open(rundir + 'emmod.out', 'rb') as outfile:
        temp = np.loadtxt(outfile, skiprows=1, unpack=True)

        # Get same x/y as requested (round to mm)
        tct = np.round(temp[0], 4) + 1j*np.round(temp[1], 4)
        tcr = np.round(rec[0], 4) + 1j*np.round(rec[1], 4)

        result = np.zeros(rec[0].shape, dtype=complex)
        for i in range(rec[0].size):
            itr = np.where(tct == tcr[i])[0]
            result[i] = (temp[3][itr] + 1j*temp[4][itr])[0]

    return result
