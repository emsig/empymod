"""Routines to create data from external modellers, for comparison purposes.

- DIPOLE1D: You must have Dipole1D installed and it must be in your system
  path; http://software.seg.org/2012/0003.

- EMmod: You must have Dipole1D installed and it must be in your system
  path; http://software.seg.org/2015/0001.

- Green3D: You must have Green3D installed (for which you need to be a member
  of the CEMI consortium). The following files must be in a folder
  `empymod/tests/green3d`: `green3d.m`, `grint.mexa64`,
  `grint.mexw64`,`normal.mexa64`, and `normal.mexw64`. Furthermore, you need
  Matlab.

Tested only on Linux (Ubuntu 16.04 LTS, x86_64).

"""
import os
import subprocess
import numpy as np
from os.path import join, dirname


class ChDir(object):
    """Step into a directory temporarily.

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
    """Run model with green3d"""

    # Execution directory
    rundir = join(dirname(__file__), 'green3d/')

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
                      '#!/bin/bash\n\n'
                      'matlab -nodesktop -nosplash -r "[e, h] = green3d('
                      '[' + ','.join(map(str, freq))+'], '
                      '[' + ','.join(map(str, depth[1:] -
                                         np.r_[0, depth[1:-1]])) + '], '
                      '[' + ','.join(map(str, 1/res[1:])) + '], '
                      '[' + ','.join(map(str, aniso[1:])) + '], '
                      '[' + ','.join(map(str, rec[0].ravel())) + '], '
                      '[' + ','.join(map(str, rec[1].ravel())) + '], '
                      '[' + ','.join(map(str, np.ones(np.size(rec[0])) *
                                     rec[2])) + '], '
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

        return Ex, Ey, Ez, Hx, Hy, Hz
