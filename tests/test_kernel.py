import numpy as np
from os.path import join, dirname
from numpy.testing import assert_allclose
from numexpr import evaluate as use_ne_eval

from empymod import kernel

# No input checks are carried out in kernel, by design. Input checks are
# carried out in model/utils, not in the core functions kernel/transform.
# Rubbish in, rubbish out. So we also do not check these functions for wrong
# inputs. These are regressions test, ensure status quo. Each test checks all
# possibility of that given function.

# Load required data
# Data generated with create_empymod.py
DATAEMPYMOD = np.load(join(dirname(__file__), 'data_empymod.npz'))
# Data generated with create_kernel.py
DATAKERNEL = np.load(join(dirname(__file__), 'data_kernel.npz'))


def test_wavenumber():                                          # 1. wavenumber
    dat = DATAKERNEL['wave'][()]
    for key, val in dat.items():
        out = kernel.wavenumber(ab=val[0], msrc=val[1], mrec=val[2], **val[3])
        assert_allclose(out[0], val[4][0])
        assert_allclose(out[1], val[4][1])
        assert_allclose(out[2], val[4][2])


def test_greenfct():                                              # 2. greenfct
    dat = DATAKERNEL['green'][()]
    for key, val in dat.items():
        for i in [3, 5, 7]:
            ab = val[0]
            msrc = val[1]
            mrec = val[2]
            out = kernel.greenfct(ab=ab, msrc=msrc, mrec=mrec, **val[i])
            assert_allclose(out[0], val[i+1][0])
            assert_allclose(out[1], val[i+1][1])
            val[i]['use_ne_eval'] = use_ne_eval
            out = kernel.greenfct(ab=ab, msrc=msrc, mrec=mrec, **val[i])
            assert_allclose(out[0], val[i+1][0])
            assert_allclose(out[1], val[i+1][1])


def test_reflections():                                        # 3. reflections
    dat = DATAKERNEL['refl'][()]
    for key, val in dat.items():
        Rp, Rm = kernel.reflections(**val[0])
        assert_allclose(Rp, val[1])
        assert_allclose(Rm, val[2])
        val[0]['use_ne_eval'] = use_ne_eval
        Rp, Rm = kernel.reflections(**val[0])
        assert_allclose(Rp, val[1])
        assert_allclose(Rm, val[2])


def test_fields():                                                  # 4. fields
    dat = DATAKERNEL['fields'][()]
    for key, val in dat.items():
        for i in [2, 4, 6, 8, 10]:
            ab = val[0]
            TM = val[1]
            Pu, Pd = kernel.fields(ab=ab, TM=TM, **val[i])
            assert_allclose(Pu, val[i+1][0])
            assert_allclose(Pd, val[i+1][1])
            val[i]['use_ne_eval'] = use_ne_eval
            Pu, Pd = kernel.fields(ab=ab, TM=TM, **val[i])
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
    for key in hs:
        # Get halfspace
        hs_res = kernel.halfspace(**hs[key])
        # Check
        assert_allclose(hs_res, hsres[key])
