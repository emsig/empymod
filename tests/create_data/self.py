r"""Create data for
   - test_model:: test_self, test_gpr, and test_dipole_k
   - test_kernel:: test_fullspace and test_halfspace."""
import numpy as np
from copy import deepcopy as dc
from scipy.constants import epsilon_0, mu_0
from empymod.model import bipole, dipole_k, gpr
from empymod.kernel import halfspace, fullspace


# # A -- BIPOLE (SELF) # #

# Test 1: EE; multiple src & rec
inp1 = {'src': [[-200, 200], [300, -300], [-100, -200], [-200, -100],
                [20, 40], [40, 20]],
        'rec': [[5000, 5200], [5100, 5350], [1000, 2000], [1010, 2020],
                [200, 500], [220, 520]],
        'depth': [0, 300, 500],
        'res': [2e14, 0.3, 2, 5],
        'freqtime': 1,
        'signal': None,
        'aniso': [1, 1, 2, 4],
        'epermH': [1, 1, 9, 15],
        'epermV': [1, 1, 20, 8],
        'mpermH': [1, 1, 1, 1],
        'mpermV': [1, 1, 5, 8],
        'msrc': False,
        'srcpts': 5,
        'mrec': False,
        'recpts': 5,
        'strength': 100,
        'xdirect': True,
        'ht': 'fht',
        'htarg': 'key_51_2012',
        'ft': 'sin',
        'ftarg': None,
        'opt': 'spline',
        'loop': None,
        'verb': 0}
EM1 = bipole(**inp1)

# Test 2: ME; multiple src-z (horizontal) & rec-z
inp2 = {'src': [[-30, 10], [40, -20], [-15, -10], [-30, -20], [3, 5], [3, 5]],
        'rec': [[400, 420], [410, 400], [50, 20], [60, 20], [30, 40],
                [22, 52]],
        'depth': [0, 25, 40],
        'res': [2e14, 10, 20, 5],
        'freqtime': 1000,
        'signal': None,
        'aniso': [2, 2, 1, 3],
        'epermH': [1, 50, 9, 15],
        'epermV': [10, 1, 20, 8],
        'mpermH': [1, 20, 1, 1],
        'mpermV': [10, 1, 5, 8],
        'msrc': False,
        'srcpts': 5,
        'mrec': True,
        'recpts': 5,
        'strength': 0,
        'xdirect': True,
        'ht': 'fht',
        'htarg': 'key_201_2009',
        'ft': 'sin',
        'ftarg': None,
        'opt': 'parallel',
        'loop': None,
        'verb': 0}
EM2 = bipole(**inp2)

# Test 3: EM; one src-z & multiple rec-z (horizontal)
inp3 = {'src': [[0, 1000], [100, 1200], [0, 0], [0, 100], 10, 10],
        'rec': [[50000, 60000], [51000, 61000], [0, 100], [0, 200], 20, 20],
        'depth': [0, 3000, 5000],
        'res': [2e14, .5, 30, 50],
        'freqtime': .01,
        'signal': None,
        'aniso': [3, 1, 1, 5],
        'epermH': [20, 1, 5, 10],
        'epermV': [12, 31, 40, 15],
        'mpermH': [11, 20, 1, 12],
        'mpermV': [1, 12, 15, 82],
        'msrc': True,
        'srcpts': 5,
        'mrec': False,
        'recpts': 5,
        'strength': 0,
        'xdirect': False,
        'ht': 'qwe',
        'htarg': [1e-10, 1e-20, 21, 40, 40],
        'ft': 'sin',
        'ftarg': None,
        'opt': 'spline',
        'loop': 'freq',
        'verb': 0}
EM3 = bipole(**inp3)

# Test 4: EM; one src & rec-z; time
inp4 = {'src': [0, 100, 0, 0, 10, 10],
        'rec': [5000, 5100, 100, 200, 20, 20],
        'depth': [0, 300, 500],
        'res': [2e14, .5, 30, 50],
        'freqtime': 1,
        'signal': 0,
        'aniso': [3, 1, 1, 5],
        'epermH': None,
        'epermV': None,
        'mpermH': None,
        'mpermV': None,
        'msrc': True,
        'srcpts': 5,
        'mrec': True,
        'recpts': 5,
        'strength': 0,
        'xdirect': True,
        'ht': 'fht',
        'htarg': None,
        'ft': 'fftlog',
        'ftarg': None,
        'opt': None,
        'loop': None,
        'verb': 0}
EM4 = bipole(**inp4)

# # B -- WAVENUMBER # #
winp = {'src': [330, -200, 500],
        'rec': [3000, 1000, 600],
        'depth': [0, 550],
        'res': [1e12, 5.55, 11],
        'freq': 3.33,
        'wavenumber': np.logspace(-3.6, -3.4, 10),
        'ab': 25,
        'aniso': [1, 2, 1.5],
        'epermH': [1, 50, 10],
        'epermV': [80, 20, 1],
        'mpermH': [1, 20, 50],
        'mpermV': [1, 30, 4],
        'verb': 0}
PJ0, PJ1 = dipole_k(**winp)

# # C -- FULLSPACE # #
# More or less random values, to test a wide range of models.
# src fixed at [0, 0, 0]; Never possible to test all combinations...
pab = [11, 12, 13, 14, 15, 16, 21, 22, 23, 24, 25, 26, 31, 32, 33, 34, 35, 41,
       42, 43, 44, 45, 46, 51, 52, 53, 54, 55, 56, 61, 62, 64, 65, 66]
prec = [[100000, 0, 500], [10000, 0, 400], [1000, 0, 300], [100, 0, 100],
        [10, 0, 10], [1, 0, 1], [1, 0, -1], [10, 0, -10], [100, 0, -100],
        [1000, 0, -300], [10000, 0, -400], [100000, 100, -500], [0, 100000, 0],
        [0, 10000, 0], [0, 1000, 500], [0, 100, 0], [10, 10, 0], [0, 1, 0],
        [0, 1, 0], [0, 10, 0], [0, 100, 500], [100, 1000, 300], [0, 10000, 0],
        [0, 100000, 0], [-1, 0, 0], [-10, 0, 500], [-100, 100, 300],
        [-1000, 0, 0], [0, -1, 0], [0, -10, 0], [-100, -100, -500],
        [0, -1000, 0], [0, -10000, 0], [50, 50, 500]]
pfreq = [0.01, 0.1, 1, 10, 100, 1000, 1000, 100, 10, 1, 0.1, 0.01, 0.01, 0.1,
         1, 10, 100, 1000, 1000, 100, 10, 1, 0.1, 0.01, 1, 10, 1, 1, 100, 10,
         1, 0.1, 0.01, 1]
pres = [10, 3, 3, 20, 4, .004, 300, 20, 1, 100, 1000, 100, 10, 10, 10, 10, 10,
        10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10]
paniso = [1, 50, 1, 1, 2, 3, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
          1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
pepH = [1, 1, 50, 100, 1, 50, 1, 50, 1, 50, 1, 50, 1, 50, 1, 50, 1, 50, 1, 50,
        1, 50, 1, 50, 1, 50, 1, 50, 1, 50, 1, 50, 1, 50]
pepV = [50, 100, 25, 1, 25, 1, 5, 1, 25, 1, 25, 1, 25, 1, 25, 1, 25, 1, 25, 1,
        25, 1, 25, 1, 25, 1, 25, 1, 25, 1, 25, 1, 25, 1]
pmpH = [67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84,
        85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100]
pmpV = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
        21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34]
fs = dict()
fsbp = dict()
fs_res = dict()
for i in range(34):
    ab = pab[i]
    rec = prec[i]
    freq = pfreq[i]
    res = pres[i]
    aniso = paniso[i]
    epermH = pepH[i]
    epermV = pepV[i]
    mpermH = pmpH[i]
    mpermV = pmpV[i]
    etaH = np.array([1/res + 2j*np.pi*freq*epermH*epsilon_0])
    etaV = np.array([1/(res*aniso**2) + 2j*np.pi*freq*epermV*epsilon_0])
    zetaH = np.array([2j*np.pi*freq*mpermH*mu_0])
    zetaV = np.array([2j*np.pi*freq*mpermV*mu_0])
    off = np.sqrt(rec[0]**2 + rec[1]**2)
    angle = np.arctan2(rec[1], rec[0])
    srcazm = 0
    srcdip = 0
    if ab % 10 in [3, 6]:
        srcdip = 90
    elif ab % 10 in [2, 5]:
        srcazm = 90
    recazm = 0
    recdip = 0
    if ab // 10 in [3, 6]:
        recdip = 90
    elif ab // 10 in [2, 5]:
        recazm = 90
    msrc = ab % 10 > 3
    mrec = ab // 10 > 3
    if mrec:
        if msrc:
            ab -= 33
        else:
            ab = ab % 10*10 + ab // 10

    # Collect dict for fullspace
    fs[str(pab[i])] = {'off': off,
                       'angle': angle,
                       'zsrc': 0,
                       'zrec': rec[2],
                       'etaH': etaH,
                       'etaV': etaV,
                       'zetaH': zetaH,
                       'zetaV': zetaV,
                       'ab': ab,
                       'msrc': msrc,
                       'mrec': mrec}

    # Collect dict for bipole
    fsbp[str(pab[i])] = {'src': [0, 0, 0, srcazm, srcdip],
                         'rec': [rec[0], rec[1], rec[2], recazm, recdip],
                         'depth': 2e14,
                         'res': [res, res+1e-10],
                         'freqtime': freq,
                         'signal': None,
                         'aniso': [aniso, aniso],
                         'epermH': [epermH, epermH],
                         'epermV': [epermV, epermV],
                         'mpermH': [mpermH, mpermH],
                         'mpermV': [mpermV, mpermV],
                         'msrc': msrc,
                         'srcpts': 1,
                         'mrec': mrec,
                         'recpts': 1,
                         'strength': 0,
                         'xdirect': False,
                         'ht': 'fht',
                         'htarg': None,
                         'ft': None,
                         'ftarg': None,
                         'opt': None,
                         'loop': None,
                         'verb': 0}

    # Get result for fullspace
    fs_res[str(pab[i])] = fullspace(**fs[str(pab[i])])

# # D -- HALFSPACE # #
# More or less random values, to test a wide range of models.
# src fixed at [0, 0, 100]; Never possible to test all combinations...
# halfspace is only implemented for electric sources and receivers so far,
# and for the diffusive approximation (low freq).
pab = [11, 12, 13, 21, 22, 23, 31, 32, 33]
prec = [[10000, -300, 500], [5000, 200, 400], [1000, 0, 300], [100, 500, 500],
        [1000, 200, 300], [0, 2000, 200], [3000, 0, 300], [10, 1000, 10],
        [100, 6000, 200]]
pres = [10, 3, 3, 3, 4, .004, 300, 20, 1]
paniso = [1, 5, 1, 3, 2, 3, 1, 1, 1]
# this is freq or time; for diffusive approximation, we must use low freqs
# or late time, according to signal
pfreq = [0.1, 5, 6, 7, 8, 9, 10, 1, 0.1]
signal = [None, 1, 1, 0, -1, 1, 0, None, None]
hs = dict()
hsbp = dict()
hs_res = dict()
for i in range(9):
    ab = pab[i]
    rec = prec[i]
    freq = pfreq[i]
    res = pres[i]
    aniso = paniso[i]
    off = np.sqrt(rec[0]**2 + rec[1]**2)
    angle = np.arctan2(rec[1], rec[0])
    srcazm = 0
    srcdip = 0
    if ab % 10 in [3, 6]:
        srcdip = 90
    elif ab % 10 in [2, 5]:
        srcazm = 90
    recazm = 0
    recdip = 0
    if ab // 10 in [3, ]:
        recdip = 90
    elif ab // 10 in [2, ]:
        recazm = 90

    c = 299792458              # Speed of light m/s
    mu_0 = 4e-7*np.pi          # Magn. permeability of free space [H/m]
    epsilon_0 = 1./(mu_0*c*c)  # Elec. permittivity of free space [F/m]
    if signal[i] is None:  # Frequency domain
        sval = 2j*np.pi*freq
    else:                  # Time domain
        sval = freq
    iwf = 2j*np.pi*freq*epsilon_0

    # Collect dict for halfspace
    hs[str(pab[i])] = {'off': off,
                       'angle': angle,
                       'zsrc': 100,
                       'zrec': rec[2],
                       'etaH': np.atleast_2d(1/res + iwf),
                       'etaV': np.atleast_2d(1/(res*aniso*aniso) + iwf),
                       'freqtime': np.atleast_2d(sval),
                       'signal': signal[i],
                       'ab': ab,
                       'solution': 'dhs'}

    # Collect dict for bipole
    hsbp[str(pab[i])] = {'src': [0, 0, 100, srcazm, srcdip],
                         'rec': [rec[0], rec[1], rec[2], recazm, recdip],
                         'depth': 0,
                         'res': [2e14, res],
                         'freqtime': freq,
                         'signal': signal[i],
                         'aniso': [1, aniso],
                         'epermH': None,
                         'epermV': None,
                         'mpermH': None,
                         'mpermV': None,
                         'msrc': False,
                         'srcpts': 1,
                         'mrec': False,
                         'recpts': 1,
                         'strength': 0,
                         'xdirect': False,
                         'ht': 'fht',
                         'htarg': None,
                         'ft': 'sin',
                         'ftarg': None,
                         'opt': None,
                         'loop': None,
                         'verb': 0}

    # Get result for halfspace
    hs_res[str(pab[i])] = halfspace(**hs[str(pab[i])])

# # E -- GPR # #
igpr = {'src': [[0, 0], [0, 1], 0.0000001],
        'rec': [[2, 3], [0, 0], 0.5],
        'depth': [0, 1],
        'res': [1e23, 200, 20],
        'freqtime': np.arange(1, 81)*1e-9,
        'cf': 250e6,
        'ab': 11,
        'gain': 3,
        'aniso': None,
        'epermH': [1, 9, 15],
        'epermV': [1, 9, 15],
        'mpermH': None,
        'mpermV': None,
        'xdirect': True,
        'ht': 'fht',
        'htarg': ['key_201_2009', ''],
        'opt': None,
        'loop': None,
        'verb': 3}
igpr2a = dc(igpr)
igpr2a['src'] = [0, 1, 0.0000001]
igpr2b = dc(igpr)
igpr2b['rec'] = [2, 0, 0.5]
ogpr = gpr(**igpr)

# # F -- Store data # #
np.savez_compressed('../data/empymod.npz',
                    out1={'inp': inp1, 'EM': EM1},
                    out2={'inp': inp2, 'EM': EM2},
                    out3={'inp': inp3, 'EM': EM3},
                    out4={'inp': inp4, 'EM': EM4},
                    wout={'inp': winp, 'PJ0': PJ0, 'PJ1': PJ1},
                    fs=fs, fsbp=fsbp, fsres=fs_res,
                    hs=hs, hsbp=hsbp, hsres=hs_res,
                    gprout={'inp': igpr, 'GPR': ogpr,
                            'inp2a': igpr2a, 'inp2b': igpr2b})
