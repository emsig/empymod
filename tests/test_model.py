# model. Status: 9/14
import pytest
import numpy as np
from os.path import join, dirname
from numpy.testing import assert_allclose
from scipy.constants import epsilon_0, mu_0

# Test 1: Import main modelling routines from empymod directly to ensure they
#         are in the __init__.py-file.
from empymod import bipole, dipole, frequency, time
# Import rest from model
from empymod.model import wavenumber, fem, tem  # gpr
from empymod.kernel import fullspace, halfspace

# These are kind of macro-tests, as they check the final results.
# I try to use different parameters for each test, to cover a wide range of
# possibilities. It won't be possible to check all the possibilities though.
# Add tests when issues arise!

# This approach for the modeller-comparisons:
#
# # Different Sets of Sources
# dipSources = [
#         [0, 0, 400],
#         [100, -100, 400]
#         ]
#
# # Different Sets of Receivers
# dipReceivers = [
#         [1000, 0, 500],
#         [1000, 1000, 1000]
#         ]
#
# # Different Sets of Models
# simpleModels = [
#         {'depth': 0, 'res': [1e12, 10], 'aniso': [1, 1]},
#         {'depth': [0, 500], 'res': [1e12, 0.3, 10], 'aniso': [1, 1, 2]}
#         ]
#
#
# # Create different types of surveys
# class SimpleDipoleSurvey:
#     @pytest.mark.parametrize("src", dipSources)
#     @pytest.mark.parametrize("rec", dipReceivers)
#     @pytest.mark.parametrize("model", simpleModels)
#     def test_single(self, src, rec, model):
#         self.do(src, rec, model)


class TestBipole:                                                   # 1. bipole
    # => Main and most important checks/comparisons

    # test freq, time (all signals)
    # test loop-options
    # test opt-options
    # test integration points
    # test source strength
    # test ht args
    # test ft args

    # Comparison to analytical fullspace solution               # 1.1 fullspace
    # More or less random values, to test a wide range of models.
    # src fixed at [0, 0, 0]; Never possible to test all combinations...
    sp1 = ("ab", "rec", "freq", "res", "aniso", "epH", "epV", "mpH", "mpV")
    vp1 = [(11, [100000, 0, 500], 0.01, 10, 1, 1, 50, 67, 1),
           (12, [10000, 0, 400], 0.1, 3, 50, 1, 100, 68, 2),
           (13, [1000, 0, 300], 1, 3, 1, 50, 25, 69, 3),
           (14, [100, 0, 100], 10, 20, 1, 100, 1, 70, 4),
           (15, [10, 0, 10], 100, 4, 2, 1, 25, 71, 5),
           (16, [1, 0, 1], 1000, .004, 3, 50, 1, 72, 6),
           (21, [1, 0, -1], 1000, 300, 1, 1, 25, 73, 7),
           (22, [10, 0, -10], 100, 20, 1, 50, 1, 74, 8),
           (23, [100, 0, -100], 10, 1, 1, 1, 25, 75, 9),
           (24, [1000, 0, -300], 1, 100, 1, 50, 1, 76, 10),
           (25, [10000, 0, -400], 0.1, 1000, 1, 1, 25, 77, 11),
           (26, [100000, 100, -500], 0.01, 100, 2, 50, 1, 78, 12),
           (31, [0, 100000, 0], 0.01, 10, 1, 1, 25, 79, 13),
           (32, [0, 10000, 0], 0.1, 10, 1, 50, 1, 80, 14),
           (33, [0, 1000, 500], 1, 10, 1, 1, 25, 81, 15),
           (34, [0, 100, 0], 10, 10, 1, 50, 1, 82, 16),
           (35, [10, 10, 0], 100, 10, 1, 1, 25, 83, 17),
           (41, [0, 1, 0], 1000, 10, 1, 50, 1, 84, 18),
           (42, [0, 1, 0], 1000, 10, 1, 1, 25, 85, 19),
           (43, [0, 10, 0], 100, 10, 1, 50, 1, 86, 20),
           (44, [0, 100, 500], 10, 10, 1, 1, 25, 87, 21),
           (45, [100, 1000, 300], 1, 10, 1, 50, 1, 88, 22),
           (46, [0, 10000, 0], 0.1, 10, 1, 1, 25, 89, 23),
           (51, [0, 100000, 0], 0.01, 10, 1, 50, 1, 90, 24),
           (52, [-1, 0, 0], 1, 10, 1, 1, 25, 91, 25),
           (53, [-10, 0, 500], 10, 10, 1, 50, 1, 92, 26),
           (54, [-100, 100, 300], 1, 10, 1, 1, 25, 93, 27),
           (55, [-1000, 0, 0], 1, 10, 1, 50, 1, 94, 28),
           (56, [0, -1, 0], 100, 10, 1, 1, 25, 95, 29),
           (61, [0, -10, 0], 10, 10, 1, 50, 1, 96, 30),
           (62, [-100, -100, -500], 1, 10, 1, 1, 25, 97, 31),
           (64, [0, -1000, 0], 0.1, 10, 1, 50, 1, 98, 32),
           (65, [0, -10000, 0], 0.01, 10, 1, 1, 25, 99, 33),
           (66, [50, 50, 500], 1, 10, 1, 50, 1, 100, 34)]

    @pytest.mark.parametrize(sp1, vp1)
    def test_fullspace(self, ab, rec, freq, res, aniso, epH, epV, mpH, mpV):
        # Calculate required parameters
        eH = np.array([1/res + 2j*np.pi*freq*epH*epsilon_0])
        eV = np.array([1/(res*aniso**2) + 2j*np.pi*freq*epV*epsilon_0])
        zH = np.array([2j*np.pi*freq*mpH*mu_0])
        zV = np.array([2j*np.pi*freq*mpV*mu_0])
        off = np.sqrt(rec[0]**2 + rec[1]**2)
        angle = np.arctan2(rec[1], rec[0])
        zrec = rec[2]
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

        # Get fullspace
        fs_res = fullspace(off, angle, 0, zrec, eH, eV, zH, zV, ab, msrc, mrec)

        # Get bipole
        bip_res = bipole([0, 0, 0, srcazm, srcdip],
                         [rec[0], rec[1], zrec, recazm, recdip], 1e20,
                         [res, res+1e-10], freq, None, [aniso, aniso],
                         [epH, epH], [epV, epV], [mpH, mpH], [mpV, mpV], msrc,
                         1, mrec, 1, 0, False, 'fht',
                         None, None, None, None, None, 0)

        # Check
        assert_allclose(fs_res, bip_res)

    # Comparison to analytical halfspace solution               # 1.2 halfspace
    # More or less random values, to test a wide range of models.
    # src fixed at [0, 0, 0]; Never possible to test all combinations...
    # halfspace is only implemented for electric sources and receivers so far,
    # and for the diffusive approximation (low freq).
    sp2 = ("ab", "rec", "freq", "res", "aniso")
    vp2 = [(11, [10000, -300, 500], 0.01, 10, 1),
           (12, [5000, 200, 400], 0.1, 3, 5),
           (13, [1000, 0, 300], 1, 3, 1),
           (21, [100, 500, 500], 1, 3, 5),
           (22, [1000, 200, 300], 1, 4, 2),
           (23, [0, 2000, 200], 0.01, .004, 3),
           (31, [3000, 0, 300], 0.1, 300, 1),
           (32, [10, 1000, 10], 1, 20, 1),
           (33, [100, 6000, 200], 0.1, 1, 1)]

    @pytest.mark.parametrize(sp2, vp2)
    def test_halfspace(self, ab, rec, freq, res, aniso):
        # Calculate required parameters
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
        msrc = ab % 10 > 3

        # Get fullspace
        hs_res = halfspace(rec[0], rec[1], 100, rec[2], res, freq, aniso, ab)

        # Get bipole
        bip_res = bipole([0, 0, 100, srcazm, srcdip],
                         [rec[0], rec[1], rec[2], recazm, recdip], 0,
                         [1e20, res], freq, None, [1, aniso],
                         None, None, None, None, msrc, 1, False, 1, 0, False,
                         'fht', None, None, None, None, None, 0)

        # Check
        assert_allclose(hs_res, bip_res)

    # 1.3. Comparison to EMmod
    # General tests, as in Comparing.ipynb

    # 1.4. Comparison to DIPOLE1D
    # Test finite length bipoles, rotated

    # 1.5. Comparison to Green3D
    # Test a few anisotropic cases

    # 1.6 Comparison to self
    # Test 4 bipole cases (EE, ME, EM, MM)


def test_dipole():                                                  # 2. dipole
    # As this is a shortcut, just run one test to ensure
    # it is equivalent to bipole.
    src = [5000, 1000, -200]
    rec = [0, 0, 1200]
    model = {'depth': [100, 1000], 'res': [2, 0.3, 100], 'aniso': [2, .5, 2]}
    f = 0.01
    # v  dipole : ab = 26
    # \> bipole : src-dip = 90, rec-azimuth=90, msrc=True
    dip_res = dipole(src, rec, freqtime=f, ab=26, verb=0, **model)
    bip_res = bipole([src[0], src[1], src[2], 0, 90],
                     [rec[0], rec[1], rec[2], 90, 0], msrc=True, freqtime=f,
                     verb=0, **model)
    assert_allclose(dip_res, bip_res)

# 3. gpr (Check it remains as in paper)


def test_wavenumber():                                          # 4. wavenumber
    # This is like `frequency`, without the Hankel transform. We just run a
    # test here, to check that it remains the status quo.
    model = {'src': [330, -200, 500],
             'rec': [3000, 1000, 600],
             'depth': [0, 550],
             'res': [1e12, 5.55, 11],
             'freq': 3.33,
             'wavenumber': np.logspace(-3.6, -3.4, 10),
             'ab': 52,
             'aniso': [1, 2, 1.5],
             'epermH': [1, 50, 10],
             'epermV': [80, 20, 1],
             'mpermH': [1, 20, 50],
             'mpermV': [1, 30, 4],
             'xdirect': True,
             'verb': 0}

    # PJ0 Result
    PJ0 = np.array([9.87407175e-10 - 6.34617396e-10j,
                    1.19134463e-09 - 6.68558766e-10j,
                    1.43235285e-09 - 7.00465477e-10j,
                    1.71674062e-09 - 7.29194616e-10j,
                    2.05184912e-09 - 7.53331115e-10j,
                    2.44621653e-09 - 7.71133632e-10j,
                    2.90976830e-09 - 7.80470046e-10j,
                    3.45403688e-09 - 7.78740522e-10j,
                    4.09241513e-09 - 7.62785619e-10j,
                    4.84044858e-09 - 7.28776375e-10j])

    # PJ1 Result
    PJ1 = np.array([-1.97481435e-09 + 1.26923479e-09j,
                    -2.38268927e-09 + 1.33711753e-09j,
                    -2.86470571e-09 + 1.40093095e-09j,
                    -3.43348125e-09 + 1.45838923e-09j,
                    -4.10369824e-09 + 1.50666223e-09j,
                    -4.89243305e-09 + 1.54226726e-09j,
                    -5.81953661e-09 + 1.56094009e-09j,
                    -6.90807375e-09 + 1.55748104e-09j,
                    -8.18483027e-09 + 1.52557124e-09j,
                    -9.68089716e-09 + 1.45755275e-09j])

    w_res0, w_res1 = wavenumber(**model)

    assert_allclose(w_res0, PJ0)
    assert_allclose(w_res1, PJ1)


def test_frequency():                                            # 5. frequency
    # As this is a shortcut, just run one test to ensure
    # it is equivalent to dipole with signal=None.
    src = [100, -100, 400]
    rec = [1000, 1000, 1000]
    model = {'depth': [0, 500], 'res': [1e12, 0.3, 10], 'aniso': [1, 1, 2]}
    f = 1
    ab = 45
    f_res = frequency(src, rec, freq=f, ab=ab, verb=0, **model)
    d_res = dipole(src, rec, freqtime=f, ab=ab, verb=0, **model)
    assert_allclose(f_res, d_res)


def test_time():                                                      # 6. time
    # As this is a shortcut, just run one test to ensure
    # it is equivalent to dipole with signal!=None.
    src = [-100, 300, 600]
    rec = [1000, -500, 400]
    model = {'depth': [-100, 600], 'res': [1e12, 3, 1], 'aniso': [1, 2, 3]}
    t = 10
    ab = 51
    signal = -1
    ft = 'fftlog'
    t_res = time(src, rec, time=t, signal=signal, ab=ab, ft=ft, verb=0,
                 **model)
    d_res = dipole(src, rec, freqtime=t, signal=signal, ab=ab, ft=ft,
                   verb=0, **model)
    assert_allclose(t_res, d_res)


def test_fem():                                                        # 7. fem
    # Just ensure functionality stays the same, with one example.
    # Data generated with create_data/create_fem_tem.py [24/01/2017]

    # Load data
    data = np.load(join(dirname(__file__), 'data_fem_tem.npz'))
    data1 = data['out1'][()]
    data2 = data['out2'][()]
    data3 = data['out3'][()]

    # Normal case: no looping
    fEM, kcount, _ = fem(**data2['inp'])
    assert_allclose(fEM, data2['EM'])
    assert kcount == data2['kcount']

    # Normal case: loop over frequencies
    data2['inp']['loop_freq'] = True
    fEM, kcount, _ = fem(**data2['inp'])
    assert_allclose(fEM, data2['EM'])
    assert kcount == data2['inp']['freq'].size

    # Normal case: loop over offsets
    data2['inp']['loop_off'] = True
    data2['inp']['loop_freq'] = False
    fEM, kcount, _ = fem(**data2['inp'])
    assert_allclose(fEM, data2['EM'])
    assert kcount == data2['inp']['off'].size

    # Fullspace
    fEM, kcount, _ = fem(**data1['inp'])
    assert_allclose(fEM, data1['EM'])
    assert kcount == data1['kcount']

    # 36 (=> zeros)
    fEM, kcount, _ = fem(**data3['inp'])
    assert_allclose(fEM, data3['EM'])
    assert kcount == data3['kcount']


def test_tem():                                                        # 8. tem
    # Just ensure functionality stays the same, with one example.
    # Data generated with create_data/create_fem_tem.py [24/01/2017]

    # Load data
    data = np.load(join(dirname(__file__), 'data_fem_tem.npz'))
    data4 = data['out4'][()]
    data5 = data['out5'][()]
    data6 = data['out6'][()]

    # # Signal = 0
    tEM, _ = tem(**data4['inp'])
    assert_allclose(tEM, data4['EM'])

    # Signal = 1
    tEM, _ = tem(**data5['inp'])
    assert_allclose(tEM, data5['EM'])

    # Signal = -1
    tEM, _ = tem(**data6['inp'])
    assert_allclose(tEM, data6['EM'])
