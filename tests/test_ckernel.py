import pytest
import numpy as np
from os.path import join, dirname
from numpy.testing import assert_allclose

from empymod import ckernel as kernel
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


def test_wavenumber():                                      # 1. wavenumber
    wavenumber = kernel.wavenumber

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


def test_greenfct():                                          # 2. greenfct
    greenfct = kernel.greenfct

    dat = DATAKERNEL['green'][()]
    for _, val in dat.items():
        for i in [3, 5, 7]:
            ab = val[0]
            msrc = val[1]
            mrec = val[2]
            out = greenfct(ab=ab, msrc=msrc, mrec=mrec, **val[i])
            assert_allclose(out[0], val[i+1][0])
            assert_allclose(out[1], val[i+1][1])


def test_reflections():                                    # 3. reflections
    reflections = kernel.reflections

    dat = DATAKERNEL['refl'][()]
    for _, val in dat.items():
        Rp, Rm = reflections(**val[0])
        assert_allclose(Rp, val[1])
        assert_allclose(Rm, val[2])


def test_fields():                                              # 4. fields
    fields = kernel.fields

    dat = DATAKERNEL['fields'][()]
    for _, val in dat.items():
        for i in [2, 4, 6, 8, 10]:
            ab = val[0]
            TM = val[1]
            Pu, Pd = fields(ab=ab, TM=TM, **val[i])
            assert_allclose(Pu, val[i+1][0])
            assert_allclose(Pd, val[i+1][1])
