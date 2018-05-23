import pytest
import numpy as np
from numpy.testing import assert_allclose
from scipy.constants import mu_0, epsilon_0

from empymod.scripts import tmtemod
from empymod import kernel, filters, dipole

# We only check that the summed return values in the functions in tmtemod agree
# with the corresponding functions from empymod. Nothing more. The examples are
# based on the examples in empymod/tests/create_data.

# Simple model, three frequencies, 6 layers
freq = np.array([0.003, 2.5, 1e6])
res = np.array([3, .3, 10, 4, 3, 1])
aniso = np.array([1, .5, 3, 1, 2, 1])
eperm = np.array([80, 100, 3, 8, 1, 1])
mperm = np.array([.5, 100, 30, 1, 30, 1])
etaH = 1/res + np.outer(2j*np.pi*freq, eperm*epsilon_0)
etaV = 1/(res*aniso*aniso) + np.outer(2j*np.pi*freq, eperm*epsilon_0)
zeta = np.outer(2j*np.pi*freq, mperm*mu_0)
filt = filters.key_201_2012()
lambd = filt.base/np.array([0.001, 1, 100, 10000])[:, None]
depth = np.array([-np.infty, 0, 150, 300, 500, 600, 800])


def test_dipole():
    for lay in [0, 1, 5]:  # Src/rec in first, second, and last layer
        for f in freq:  # One freq at a time
            src = [0, 0, depth[lay+1]-50]

            # Offset depending on frequency
            if f < 1:
                rec = [10000, 0, depth[lay+1]-10]
            elif f > 10:
                rec = [2, 0, depth[lay+1]-10]
            else:
                rec = [1000, 0, depth[lay+1]-10]
            inp = {'src': src, 'rec': rec, 'depth': depth[1:-1], 'res': res,
                   'freqtime': f, 'aniso': aniso, 'verb': 0}

            # empymod-version
            out = dipole(epermH=eperm, epermV=eperm, mpermH=mperm,
                         mpermV=mperm, xdirect=False, **inp)

            # empymod.scripts-version
            TM, TE = tmtemod.dipole(eperm=eperm, mperm=mperm, **inp)
            TM = TM[0] + TM[1] + TM[2] + TM[3] + TM[4]
            TE = TE[0] + TE[1] + TE[2] + TE[3] + TE[4]

            # Check
            assert_allclose(out, TM + TE, rtol=1e-5, atol=1e-50)

    # Check the 3 errors
    with pytest.raises(ValueError):  # scr/rec not in same layer
        tmtemod.dipole([0, 0, 90], [4000, 0, 180], depth[1:-1], res, 1)

    with pytest.raises(ValueError):  # more than one frequency
        tmtemod.dipole([0, 0, 90], [4000, 0, 110], depth[1:-1], res, [1, 10])

    with pytest.raises(ValueError):  # only one layer
        tmtemod.dipole([0, 0, 90], [4000, 0, 110], [], 10, 1)


def test_greenfct():
    for lay in [0, 1, 5]:  # src/rec in first, second, and last layer
        inp = {'depth': depth[:-1], 'lambd': lambd,
               'etaH': etaH, 'etaV': etaV,
               'zetaH': zeta, 'zetaV': zeta,
               'lrec': np.array(lay), 'lsrc': np.array(lay),
               'zsrc': depth[lay+1]-50, 'zrec': depth[lay+1]-10}

        # empymod-version
        out1, out2 = kernel.greenfct(ab=11, xdirect=False, msrc=False,
                                     mrec=False, use_ne_eval=False, **inp)

        # empymod.scripts-version
        TM, TE = tmtemod.greenfct(**inp)
        TM = TM[0] + TM[1] + TM[2] + TM[3] + TM[4]
        TE = TE[0] + TE[1] + TE[2] + TE[3] + TE[4]

        # Check
        assert_allclose(out1, TM, atol=1e-100)
        assert_allclose(out2, TE)


def test_fields():
    Gam = np.sqrt((etaH/etaV)[:, None, :, None] *
                  (lambd**2)[None, :, None, :] + (zeta**2)[:, None, :, None])

    for lay in [0, 1, 5]:  # Src/rec in first, second, and last layer

        inp1 = {'depth': depth[:-1], 'e_zH': etaH, 'Gam': Gam,
                'lrec': np.array(lay), 'lsrc': np.array(lay),
                'use_ne_eval': False}
        Rp1, Rm1 = kernel.reflections(**inp1)

        inp2 = {'depth': depth[:-1], 'Gam': Gam, 'Rp': Rp1, 'Rm': Rm1,
                'lrec': np.array(lay), 'lsrc': np.array(lay),
                'zsrc': depth[lay+1]-50}

        for TM in [True, False]:
            inp2['TM'] = TM

            # empymod-version
            out = kernel.fields(ab=11, use_ne_eval=False, **inp2)

            # empymod.scripts-version
            TMTE = tmtemod.fields(**inp2)

            # Check
            assert_allclose(out[0], TMTE[0] + TMTE[1])
            assert_allclose(out[1], TMTE[2] + TMTE[3])
