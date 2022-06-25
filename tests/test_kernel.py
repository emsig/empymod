import pytest
import numpy as np
from os.path import join, dirname
from numpy.testing import assert_allclose

from empymod import kernel
from empymod import bipole

# No input checks are carried out in kernel, by design. Input checks are
# carried out in model/utils, not in the core functions kernel/transform.
# Rubbish in, rubbish out. So we also do not check these functions for wrong
# inputs. These are regressions test, ensure status quo. Each test checks all
# possibility of that given function.

# Load required data
# Data generated with create_data/self.py
DATAEMPYMOD = np.load(join(dirname(__file__), 'data/empymod.npz'),
                      allow_pickle=True)
# Data generated with create_data/kernel.py
DATAKERNEL = np.load(join(dirname(__file__), 'data/kernel.npz'),
                     allow_pickle=True)


@pytest.mark.parametrize("njit", [True, False])
def test_wavenumber(njit):                                      # 1. wavenumber
    if njit:
        wavenumber = kernel.wavenumber
    else:
        wavenumber = kernel.wavenumber.py_func

    dat = DATAKERNEL['wave'][()]
    for _, val in dat.items():
        out = wavenumber(ab=val[0], msrc=val[1], mrec=val[2], **val[3])

        if val[0] in [11, 22, 24, 15, 33]:
            assert_allclose(out[0], val[4][0], atol=1e-100)
        else:
            assert out[0] is None

        if val[0] == 33:
            assert out[1] is None
        else:
            assert_allclose(out[1], val[4][1], atol=1e-100)

        if val[0] in [11, 22, 24, 15, 12, 21, 14, 25]:
            assert_allclose(out[2], val[4][2], atol=1e-100)
        else:
            assert out[2] is None


@pytest.mark.parametrize("njit", [True, False])
def test_greenfct(njit):                                          # 2. greenfct
    if njit:
        greenfct = kernel.greenfct
    else:
        greenfct = kernel.greenfct.py_func

    dat = DATAKERNEL['green'][()]
    for _, val in dat.items():
        for i in [3, 5, 7]:
            ab = val[0]
            msrc = val[1]
            mrec = val[2]
            out = greenfct(ab=ab, msrc=msrc, mrec=mrec, **val[i])
            assert_allclose(out[0], val[i+1][0])
            assert_allclose(out[1], val[i+1][1])


@pytest.mark.parametrize("njit", [True, False])
def test_reflections(njit):                                    # 3. reflections
    if njit:
        reflections = kernel.reflections
    else:
        reflections = kernel.reflections.py_func

    dat = DATAKERNEL['refl'][()]
    for _, val in dat.items():
        Rp, Rm = reflections(**val[0])
        assert_allclose(Rp, val[1])
        assert_allclose(Rm, val[2])


@pytest.mark.parametrize("njit", [True, False])
def test_fields(njit):                                              # 4. fields
    if njit:
        fields = kernel.fields
    else:
        fields = kernel.fields.py_func

    dat = DATAKERNEL['fields'][()]
    for _, val in dat.items():
        for i in [2, 4, 6, 8, 10]:
            ab = val[0]
            TM = val[1]
            Pu, Pd = fields(ab=ab, TM=TM, **val[i])
            assert_allclose(Pu, val[i+1][0])
            assert_allclose(Pd, val[i+1][1])


def test_angle_factor():                                      # 5. angle_factor
    dat = DATAKERNEL['angres'][()]
    for ddat in dat:
        res = kernel.angle_factor(**ddat['inp'])
        assert_allclose(res, ddat['res'])


def test_fullspace():                                            # 6. fullspace
    # Compare all to maintain status quo.
    fs = DATAEMPYMOD['fs'][()]
    fsres = DATAEMPYMOD['fsres'][()]
    for key in fs:
        # Get fullspace
        fs_res = kernel.fullspace(**fs[key])
        # Check
        assert_allclose(fs_res, fsres[key])


def test_halfspace():                                            # 7. halfspace
    # Compare all to maintain status quo.
    hs = DATAEMPYMOD['hs'][()]
    hsres = DATAEMPYMOD['hsres'][()]
    hsbp = DATAEMPYMOD['hsbp'][()]
    for key in hs:
        # Get halfspace
        hs_res = kernel.halfspace(**hs[key])
        # Check  # rtol decreased in June '22 - suddenly failed; why?
        #        # (Potentially as SciPy changed mu_0 to inexact,
        #           https://github.com/scipy/scipy/issues/11341).
        assert_allclose(hs_res, hsres[key], rtol=5e-5)

    # Additional checks - Time
    full = kernel.halfspace(**hs['21'])

    # Check halfspace = sum of split
    hs['21']['solution'] = 'dsplit'
    direct, reflect, air = kernel.halfspace(**hs['21'])
    assert_allclose(full, direct+reflect+air)

    # Check fullspace = bipole-solution
    hsbp['21']['xdirect'] = True
    hsbp['21']['depth'] = []
    hsbp['21']['res'] = hsbp['21']['res'][1]
    hsbp['21']['aniso'] = hsbp['21']['aniso'][1]
    hsbp['21']['ft'] = 'dlf'
    hs_res = bipole(**hsbp['21'])
    assert_allclose(direct, hs_res, rtol=1e-2)

    # Additional checks - Frequency
    hs['11']['solution'] = 'dfs'
    full = kernel.halfspace(**hs['11'])

    # Check halfspace = sum of split
    hs['11']['solution'] = 'dsplit'
    direct, _, _ = kernel.halfspace(**hs['11'])
    assert_allclose(full, direct)

    # Check sums of dtetm = dsplit
    hs['11']['solution'] = 'dsplit'
    direct, reflect, air = kernel.halfspace(**hs['11'])
    hs['11']['solution'] = 'dtetm'
    dTE, dTM, rTE, rTM, air2 = kernel.halfspace(**hs['11'])
    assert_allclose(direct, dTE+dTM)
    assert_allclose(reflect, rTE+rTM)
    assert_allclose(air, air2)

    # Check fullspace = bipole-solution
    hsbp['11']['xdirect'] = True
    hsbp['11']['depth'] = []
    hsbp['11']['res'] = hsbp['11']['res'][1]
    hsbp['11']['aniso'] = hsbp['11']['aniso'][1]
    hs_res = bipole(**hsbp['11'])
    assert_allclose(direct, hs_res, atol=1e-2)
