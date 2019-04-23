import numpy as np
from os.path import join, dirname
from numpy.testing import assert_allclose

# See if numexpr is installed, and if it is, if it uses VML
try:
    from numexpr import use_vml, evaluate as use_ne_eval
except ImportError:
    use_vml = False
    use_ne_eval = False


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


def test_wavenumber():                                          # 1. wavenumber
    dat = DATAKERNEL['wave'][()]
    for _, val in dat.items():
        out = kernel.wavenumber(ab=val[0], msrc=val[1], mrec=val[2], **val[3])

        if val[0] in [11, 22, 24, 15, 33]:
            assert_allclose(out[0], val[4][0])
        else:
            assert out[0] is None

        if val[0] == 33:
            assert out[1] is None
        else:
            assert_allclose(out[1], val[4][1])

        if val[0] in [11, 22, 24, 15, 12, 21, 14, 25]:
            assert_allclose(out[2], val[4][2])
        else:
            assert out[2] is None


def test_greenfct():                                              # 2. greenfct
    dat = DATAKERNEL['green'][()]
    for _, val in dat.items():
        for i in [3, 5, 7]:
            ab = val[0]
            msrc = val[1]
            mrec = val[2]
            out = kernel.greenfct(ab=ab, msrc=msrc, mrec=mrec, **val[i])
            assert_allclose(out[0], val[i+1][0])
            assert_allclose(out[1], val[i+1][1])
            if use_vml:  # Check if numexpr yields same result
                val[i]['use_ne_eval'] = use_ne_eval
                out = kernel.greenfct(ab=ab, msrc=msrc, mrec=mrec, **val[i])
                assert_allclose(out[0], val[i+1][0])
                assert_allclose(out[1], val[i+1][1])


def test_reflections():                                        # 3. reflections
    dat = DATAKERNEL['refl'][()]
    for _, val in dat.items():
        Rp, Rm = kernel.reflections(**val[0])
        assert_allclose(Rp, val[1])
        assert_allclose(Rm, val[2])
        val[0]['use_ne_eval'] = use_ne_eval
        Rp, Rm = kernel.reflections(**val[0])
        assert_allclose(Rp, val[1])
        assert_allclose(Rm, val[2])


def test_fields():                                                  # 4. fields
    dat = DATAKERNEL['fields'][()]
    for _, val in dat.items():
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
    hsbp = DATAEMPYMOD['hsbp'][()]
    for key in hs:
        # Get halfspace
        hs_res = kernel.halfspace(**hs[key])
        # Check
        assert_allclose(hs_res, hsres[key])

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
    hsbp['21']['ft'] = 'sin'
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
