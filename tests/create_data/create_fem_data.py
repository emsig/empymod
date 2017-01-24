"""Create data for test_model::test_fem."""
import pickle
import numpy as np
from copy import deepcopy
from empymod import filters
from empymod.model import fem

# Model
# res = [3, 10, 3]
# aniso = [1, 2, 3]
# mpermH = [1, 5, 10]
# mpermV = [2, 5, 5]
# epermH = [1, 80, 50]
# epermV = [1, 80, 80]
depth = np.array([-np.inf, 0., 500.])
etaH = np.array([[0.33333333 + 5.56325028e-12j, 0.10000000 + 4.45060022e-10j,
                  0.33333333 + 2.78162514e-10j],
                 [0.33333333 + 5.56325028e-11j, 0.10000000 + 4.45060022e-09j,
                  0.33333333 + 2.78162514e-09j],
                 [0.33333333 + 5.56325028e-10j, 0.10000000 + 4.45060022e-08j,
                  0.33333333 + 2.78162514e-08j]])
etaV = np.array([[0.33333333 + 5.56325028e-12j, 0.02500000 + 4.45060022e-10j,
                  0.03703704 + 4.45060022e-10j],
                 [0.33333333 + 5.56325028e-11j, 0.02500000 + 4.45060022e-09j,
                  0.03703704 + 4.45060022e-09j],
                 [0.33333333 + 5.56325028e-10j, 0.02500000 + 4.45060022e-08j,
                  0.03703704 + 4.45060022e-08j]])
zetaH = np.array([[0. + 7.89568352e-07j, 0. + 3.94784176e-06j,
                   0. + 7.89568352e-06j],
                  [0. + 7.89568352e-06j, 0. + 3.94784176e-05j,
                   0. + 7.89568352e-05j],
                  [0. + 7.89568352e-05j, 0. + 3.94784176e-04j,
                   0. + 7.89568352e-04j]])
zetaV = np.array([[0. + 1.57913670e-06j, 0. + 3.94784176e-06j,
                   0. + 3.94784176e-06j],
                  [0. + 1.57913670e-05j, 0. + 3.94784176e-05j,
                   0. + 3.94784176e-05j],
                  [0. + 1.57913670e-04j, 0. + 3.94784176e-04j,
                   0. + 3.94784176e-04j]])
freq = np.array([0.1, 1., 10.])

# src and rec
# rec = [np.arange(1,11)*500, np.arange(1,11)*100, 300]
# src = [[0, -100], [0, -200], 200]
nsrc = 2
nrec = 10
off = np.array([509.90195136, 1019.80390272, 1529.70585408, 2039.60780544,
                2549.5097568, 3059.41170816, 3569.31365951, 4079.21561087,
                4589.11756223, 5099.01951359, 670.82039325, 1170.46999107,
                1676.30546142, 2184.03296678, 2692.58240357, 3201.56211872,
                3710.79506306, 4220.1895692, 4729.69343615, 5239.27475897])
angle = np.arange(1, 21)*0.01
angle = np.array([0.19739556, 0.19739556, 0.19739556, 0.19739556, 0.19739556,
                  0.19739556, 0.19739556, 0.19739556, 0.19739556, 0.19739556,
                  0.46364761, 0.348771, 0.30288487, 0.27829966, 0.26299473,
                  0.25255428, 0.24497866, 0.2392316, 0.23472261, 0.23109067])
lsrc = np.array(1)
lrec = np.array(1)
zsrc = np.array([200])
zrec = np.array([300])

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
        'ht': 'fht',
        'htarg': (filters.key_401_2009(), None),
        'use_spline': False,
        'use_ne_eval': False,
        'msrc': True,
        'mrec': True,
        'loop_freq': False,
        'loop_off': False}

EM1, kcount1, _ = fem(**inp1)

# 3. NORMAL CASE
inp2 = deepcopy(inp1)
inp2['etaH'] = etaH
inp2['etaV'] = etaV
inp2['zetaH'] = zetaH
inp2['zetaV'] = zetaV
inp2['isfullspace'] = False
inp2['msrc'] = False
inp2['mrec'] = False

EM2, kcount2, _ = fem(**inp2)

# 3. 36/63
inp3 = deepcopy(inp2)
inp3['ab'] = 36
inp3['msrc'] = True
EM3, kcount3, _ = fem(**inp3)

# Store data
out = {'inp1': inp1, 'EM1': EM1, 'kcount1': kcount1,
       'inp2': inp2, 'EM2': EM2, 'kcount2': kcount2,
       'inp3': inp3, 'EM3': EM3, 'kcount3': kcount3}
pickle.dump(out, open('../fem_data.pck', 'xb'), -1)
