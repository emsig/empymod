r"""Create data for test_fem and test_tem."""
import numpy as np
from copy import deepcopy

from empymod import utils
from empymod.model import fem, tem


# # A -- FEM # #

# Model
freq = [0.1, 1, 10]
depth = [0, 500]
res = [3, 10, 3]
aniso = [1, 2, 3]
mpermH = [1, 5, 10]
mpermV = [2, 5, 5]
epermH = [1, 80, 50]
epermV = [1, 80, 80]
model = utils.check_model(depth, res, aniso, epermH, epermV, mpermH, mpermV,
                          True, 0)
depth, res, aniso, epermH, epermV, mpermH, mpermV, isfullspace = model
frequency = utils.check_frequency(freq, res, aniso, epermH, epermV, mpermH,
                                  mpermV, 0)
freq, etaH, etaV, zetaH, zetaV = frequency

# src and rec
rec = [np.arange(1, 11)*500, np.arange(1, 11)*100, 300]
src = [[0, -100], [0, -200], 200]
src, nsrc = utils.check_dipole(src, 'src', 0)
rec, nrec = utils.check_dipole(rec, 'rec', 0)
off, angle = utils.get_off_ang(src, rec, nsrc, nrec, 0)
lsrc, zsrc = utils.get_layer_nr(src, depth)
lrec, zrec = utils.get_layer_nr(rec, depth)

ht, htarg = utils.check_hankel('fht', None, 0)

# 1. FULLSPACE
inp1 = {'ab': 12,
        'off': off,
        'angle': angle,
        'zsrc': zsrc,
        'zrec': zrec,
        'lsrc': lsrc,
        'lrec': lrec,
        'depth': depth,
        'freq': freq,
        'etaH': np.ones((3, 3))*etaH[:, 0],
        'etaV': np.ones((3, 3))*etaV[:, 0],
        'zetaH': np.ones((3, 3))*zetaH[:, 0],
        'zetaV': np.ones((3, 3))*zetaV[:, 0],
        'xdirect': True,
        'isfullspace': True,
        'ht': ht,
        'htarg': htarg,
        'use_ne_eval': False,
        'msrc': True,
        'mrec': True,
        'loop_freq': False,
        'loop_off': False}

EM1, kcount1, _ = fem(**inp1)

# 2. NORMAL CASE
inp2 = deepcopy(inp1)
inp2['etaH'] = etaH
inp2['etaV'] = etaV
inp2['zetaH'] = zetaH
inp2['zetaV'] = zetaV
inp2['isfullspace'] = False
inp2['msrc'] = False
inp2['mrec'] = False

EM2, kcount2, _ = fem(**inp2)

# 3. NORMAL CASE; loop_freq
inp3 = deepcopy(inp2)
inp3['loop_freq'] = True
EM3, kcount3, _ = fem(**inp3)

# 4. NORMAL CASE; loop_off
inp4 = deepcopy(inp2)
inp4['loop_off'] = True
EM4, kcount4, _ = fem(**inp4)

# 5. 36/63
inp5 = deepcopy(inp2)
inp5['ab'] = 36
inp5['msrc'] = True
EM5, kcount5, _ = fem(**inp5)

# # B -- TEM # #
# Same parameters as for fem, except frequency, time, ab; plus signal

# ab = 22
time = [1, 2, 3]
time, freq, ft, ftarg = utils.check_time(time, 0, 'fftlog', None, 0)
frequency = utils.check_frequency(freq, res, aniso, epermH, epermV, mpermH,
                                  mpermV, 0)
freq, etaH, etaV, zetaH, zetaV = frequency

tinp = deepcopy(inp2)
tinp['freq'] = freq
tinp['etaH'] = etaH
tinp['etaV'] = etaV
tinp['zetaH'] = zetaH
tinp['zetaV'] = zetaV
fEM, _, _ = fem(**tinp)

# 6. TIME; SIGNAL = 0
inp6 = {'fEM': fEM, 'off': off, 'freq': freq, 'time': time, 'signal': 0, 'ft':
        ft, 'ftarg': ftarg}
EM6, _ = tem(**inp6)

# 7. TIME; SIGNAL = 1
inp7 = deepcopy(inp6)
inp7['signal'] = 1
EM7, _ = tem(**inp7)

# 8. TIME; SIGNAL = -1
_, freq1, _, _ = utils.check_time(time, 1, 'fftlog', None, 0)
_, freq2, _, _ = utils.check_time(time, -1, 'fftlog', None, 0)
frequency2 = utils.check_frequency(freq2, res, aniso, epermH, epermV, mpermH,
                                   mpermV, 0)
freq2, etaH2, etaV2, zetaH2, zetaV2 = frequency2

tinp2 = deepcopy(inp2)
tinp2['freq'] = freq2
tinp2['etaH'] = etaH2
tinp2['etaV'] = etaV2
tinp2['zetaH'] = zetaH2
tinp2['zetaV'] = zetaV2
fEM2, _, _ = fem(**tinp2)
inp8 = {'fEM': fEM2, 'off': off, 'freq': freq2, 'time': time, 'signal': -1,
        'ft': ft, 'ftarg': ftarg}
EM8, _ = tem(**inp8)

# Store data
np.savez_compressed('../data/fem_tem.npz',
                    out1={'inp': inp1, 'EM': EM1, 'kcount': kcount1},
                    out2={'inp': inp2, 'EM': EM2, 'kcount': kcount2},
                    out3={'inp': inp3, 'EM': EM3, 'kcount': kcount3},
                    out4={'inp': inp4, 'EM': EM4, 'kcount': kcount4},
                    out5={'inp': inp5, 'EM': EM5, 'kcount': kcount5},
                    out6={'inp': inp6, 'EM': EM6},
                    out7={'inp': inp7, 'EM': EM7},
                    out8={'inp': inp8, 'EM': EM8})
