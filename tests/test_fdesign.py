import os
import sys
import pytest
import numpy as np
from timeit import default_timer
from os.path import join, dirname
from numpy.testing import assert_allclose

# Optional imports
try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = False

from empymod import filters, model
from empymod.scripts import fdesign

# Load required data
# Data generated with create_data/fdesign.py
DATA = np.load(join(dirname(__file__), 'data/fdesign.npz'), allow_pickle=True)


def test_design():
    # 1. General case with various spacing and shifts
    fI = (fdesign.j0_1(5), fdesign.j1_1(5))
    dat1 = DATA['case1'][()]
    _, out1 = fdesign.design(fI=fI, verb=0, plot=0, **dat1[0])
    assert_allclose(out1[0], dat1[2][0])
    assert_allclose(out1[1], dat1[2][1], rtol=1e-3)
    assert_allclose(out1[2], dat1[2][2])

    # np.linalg(.qr) can have roundoff errors which are not deterministic,
    # which can yield different results for badly conditioned matrices. This
    # only affects the edge-cases, not the best result we are looking for.
    # However, we have to limit the following comparison; we check that at
    # least 50% are within a relative error of 0.1%.
    rate = np.sum(np.abs((out1[3] - dat1[2][3])/dat1[2][3]) < 1e-3)
    assert rate > out1[3].size/2

    # 2. Specific model with only one spacing/shift
    dat2 = DATA['case2'][()]
    _, out2 = fdesign.design(fI=fI, verb=0, plot=0, **dat2[0])
    assert_allclose(out2[0], dat2[2][0])
    assert_allclose(out2[1], dat2[2][1], rtol=1e-3)
    assert_allclose(out2[2], dat2[2][2])
    assert_allclose(out2[3], dat2[2][3], rtol=1e-3)

    # 3. Same, with only one transform
    dat2b = DATA['case3'][()]
    _, out2b = fdesign.design(fI=fI[0], verb=0, plot=0, **dat2b[0])
    assert_allclose(out2b[0], dat2b[2][0])
    assert_allclose(out2b[1], dat2b[2][1], rtol=1e-3)
    assert_allclose(out2b[2], dat2b[2][2])
    assert_allclose(out2b[3], dat2b[2][3], rtol=1e-3)

    # 4.a Maximize r
    dat4 = DATA['case4'][()]
    dat4[0]['save'] = True
    dat4[0]['name'] = 'tmpfilter'
    _, out4 = fdesign.design(fI=fI, verb=0, plot=0, **dat4[0])
    assert_allclose(out4[0], dat4[2][0])
    assert_allclose(out4[1], dat4[2][1], rtol=1e-3)
    assert_allclose(out4[2], dat4[2][2])
    assert_allclose(out4[3], dat4[2][3], rtol=1e-3)
    # Clean-up  # Should be replaced eventually by tmpdir
    os.remove('./filters/tmpfilter_base.txt')
    os.remove('./filters/tmpfilter_j0.txt')
    os.remove('./filters/tmpfilter_j1.txt')
    os.remove('./filters/tmpfilter_full.txt')

    # 4.b Without full output and all the other default inputs
    dat4[0]['full_output'] = False
    del dat4[0]['name']
    dat4[0]['finish'] = 'Wrong input'
    del dat4[0]['r']
    dat4[0]['reim'] = np.imag  # Set once to imag
    fdesign.design(fI=fI, verb=2, plot=0, **dat4[0])
    # Clean-up  # Should be replaced eventually by tmpdir
    os.remove('./filters/dlf_201_base.txt')
    os.remove('./filters/dlf_201_j0.txt')
    os.remove('./filters/dlf_201_j1.txt')

    # 5. j2 for fI
    with pytest.raises(ValueError):
        fI2 = fdesign.empy_hankel('j2', 0, 50, 100, 1)
        fdesign.design(fI=fI2, verb=0, plot=0, **dat4[0])


@pytest.mark.skipif(sys.version_info < (3, 6),
                    reason="tmpdir seems to fail for Python<3.6.")
def test_save_load_filter(tmpdir):
    # Save two filters, with and without inversion result. In test_load_filter
    # we check, if they were saved correctly
    dat1 = DATA['case1'][()]
    dat1[1].name = 'one'
    dat1[1].savename = 'one'
    dat1[1].filter_coeff = ['j0', 'j1', 'sin', 'cos']
    dat2 = DATA['case2'][()]
    dat2[1].name = 'two'
    dat2[1].savename = 'two'
    dat2[1].filter_coeff = ['j0', 'j1', 'sin', 'cos']
    fdesign.save_filter('one', dat1[1], dat1[2], path=tmpdir)
    fdesign.save_filter('one.gz', dat1[1], dat1[2], path=tmpdir)  # Check gz
    fdesign.save_filter('two', dat2[1], path=tmpdir)

    # Check the filters and inversion output saved in test_save_filter
    dat1 = DATA['case1'][()]
    dat2 = DATA['case2'][()]
    filt, out = fdesign.load_filter('one', True, path=tmpdir)
    _, _ = fdesign.load_filter('one.gz', True, path=tmpdir)  # Check gz works
    assert_allclose(filt.base, dat1[1].base)
    assert_allclose(out[0], dat1[2][0])
    assert_allclose(out[1], dat1[2][1])
    assert_allclose(out[2], dat1[2][2])
    assert_allclose(out[3], dat1[2][3])
    assert_allclose(out[4], dat1[2][4])

    filt = fdesign.load_filter('two', True, path=tmpdir)
    assert_allclose(filt.base, dat2[1].base)
    filt = fdesign.load_filter('two', path=tmpdir, filter_coeff=['bla'])
    assert_allclose(filt.base, dat2[1].base)


@pytest.mark.skipif(not plt, reason="Matplotlib not installed.")
class TestFiguresMatplotlib:

    @pytest.mark.mpl_image_compare(remove_text=True)
    def test_plot_result1(self):
        # Quick run `design` with all verb/plot on, just to check that no
        # errors occur. Actually plots are checked in test below and the other
        # tests.
        dat2 = DATA['case2'][()]
        fdesign.design(fI=fdesign.j0_1(5), verb=2, plot=2, **dat2[0])

        # plot_result for min amplitude
        dat1 = DATA['case1'][()]
        fdesign.plot_result(dat1[1], dat1[2], prntres=True)
        return plt.gcf()

    @pytest.mark.mpl_image_compare(remove_text=True)
    def test_plot_result2(self):
        # plot_result one shift several spacings
        dat5 = DATA['case5'][()]
        fdesign.plot_result(dat5[1], dat5[2])
        return plt.gcf()

    @pytest.mark.mpl_image_compare(remove_text=True)
    def test_plot_result3(self):
        # plot_result several shifts one spacing for max r
        dat6 = DATA['case6'][()]
        fdesign.plot_result(dat6[1], dat6[2])
        return plt.gcf()

    @pytest.mark.mpl_image_compare(remove_text=True, tolerance=8)
    def test_call_qc_transform_pairs1(self):
        # plot_transform_pair "normal" case
        r = np.logspace(1, 2, 50)
        fI = (fdesign.j0_1(5), fdesign.j1_1(5))
        fC = (fdesign.j0_3(5), fdesign.j1_3(5))
        fC[0].rhs = fC[0].rhs(r)
        fC[1].rhs = fC[1].rhs(r)
        fdesign._call_qc_transform_pairs(101, (0.06, 0.07, 0.01), (-1, 1, 0.3),
                                         fI, fC, r, (0, 0, 2), np.real)
        return plt.gcf()

    @pytest.mark.mpl_image_compare(remove_text=True, tolerance=6)
    def test_call_qc_transform_pairs2(self):
        # plot_transform_pair J2
        r = np.logspace(1, 2, 50)
        fI = (fdesign.j0_1(5), fdesign.j1_1(5))
        fC = fdesign.empy_hankel('j2', 950, 1000, 1, 1)
        fC.rhs = fC.rhs(r)
        fdesign._call_qc_transform_pairs(101, (0.06, 0.07, 0.01), (-1, 1, 0.3),
                                         fI, [fC, ], r, (0, 0, 2), np.imag)
        return plt.gcf()

    @pytest.mark.mpl_image_compare(remove_text=True, tolerance=6)
    def test_call_qc_transform_pairs3(self):
        # plot_transform_pair Sine/Cosine
        r = np.logspace(1, 2, 50)
        fI = (fdesign.sin_1(), fdesign.cos_1())
        fC = (fdesign.sin_2(), fdesign.cos_2())
        fC[0].rhs = fC[0].rhs(r)
        fC[1].rhs = fC[1].rhs(r)
        fdesign._call_qc_transform_pairs(101, (0.06, 0.07, 0.01), (-1, 1, 0.3),
                                         fI, fC, r, (0, 0, 2), np.imag)
        return plt.gcf()

    @pytest.mark.mpl_image_compare(remove_text=True)
    def test_plot_inversion1(self):
        # plot_inversion minimum amplitude
        f = fdesign.j0_1(5)
        filt = filters.key_201_2009()
        n = filt.base.size
        a = filt.base[-1]
        b = filt.base[-2]
        spacing = np.log(a)-np.log(b)
        shift = np.log(a)-spacing*(n//2)
        cvar = 'amp'
        r = np.logspace(0, 1.5, 100)
        f.rhs = f.rhs(r)
        k = filt.base/r[:, None]
        rhs = np.dot(f.lhs(k), filt.j0)/r
        rel_error = np.abs((rhs - f.rhs)/f.rhs)
        imin = np.where(rel_error > 0.01)[0][0]
        fdesign._plot_inversion(f, rhs, r, k, imin, spacing, shift, cvar)
        return plt.gcf()

    @pytest.mark.mpl_image_compare(remove_text=True)
    def test_plot_inversion2(self):
        # plot_inversion maximum r
        f = fdesign.empy_hankel('j2', 50, 100, 1, 1)
        filt = filters.key_201_2009()
        n = filt.base.size
        a = filt.base[-1]
        b = filt.base[-2]
        spacing = np.log(a)-np.log(b)
        shift = np.log(a)-spacing*(n//2)
        cvar = 'r'
        r = np.logspace(1, 4.5, 100)
        f.rhs = f.rhs(r)
        k = filt.base/r[:, None]
        lhs = f.lhs(k)
        rhs0 = np.dot(lhs[0], filt.j0)/r
        rhs1 = np.dot(lhs[1], filt.j1)/r**2
        rhs = rhs0 + rhs1
        rel_error = np.abs((rhs - f.rhs)/f.rhs)
        imin = np.where(rel_error > 0.01)[0][0]
        fdesign._plot_inversion(f, rhs, r, k, imin, spacing, shift, cvar)
        return plt.gcf()


@pytest.mark.skipif(plt, reason="Matplotlib is installed.")
class TestFiguresNoMatplotlib:

    def test_design(self, capsys):
        # Check it doesn't fail, message is correct, and input doesn't matter
        # Same test as first in test_design
        fI = (fdesign.j0_1(5), fdesign.j1_1(5))
        dat1 = DATA['case1'][()]
        _, _ = fdesign.design(fI=fI, verb=1, plot=2, **dat1[0])
        out, _ = capsys.readouterr()
        assert "* WARNING :: `matplotlib` is not installed, no " in out

    def test_plot_result(self, capsys):
        # Check it doesn't fail, message is correct, and input doesn't matter
        fdesign.plot_result(0, 1)
        out, _ = capsys.readouterr()
        assert "* WARNING :: `matplotlib` is not installed, no " in out

    def test_plot_inversion(self, capsys):
        # Check it doesn't fail, message is correct, and input doesn't matter
        fdesign._plot_inversion(1, 2, 3, 4, 5, 6, 7, 8)
        out, _ = capsys.readouterr()
        assert "* WARNING :: `matplotlib` is not installed, no " in out


def test_print_data(capsys):
    # Test full with min amplitude with case 2
    dat1 = DATA['case2'][()]
    fdesign.print_result(dat1[1], dat1[2])
    out, _ = capsys.readouterr()
    assert "Filter length   : 201" in out
    assert "Best filter" in out
    assert "> Min field     : 1.79577e-11" in out
    assert "> Spacing       : 0.0641" in out
    assert "> Shift         : -1.2847" in out
    assert "> Base min/max  : 4.552335e-04 / 1.682246e+02" in out

    # Test full with max r with case 3
    dat2 = DATA['case4'][()]
    fdesign.print_result(dat2[1], dat2[2])
    out, _ = capsys.readouterr()
    assert "Filter length   : 201" in out
    assert "Best filter" in out
    assert "> Max r         : 21.5443" in out
    assert "> Spacing       : 0.0641" in out
    assert "> Shift         : -1.2847" in out
    assert "> Base min/max  : 4.552335e-04 / 1.682246e+02" in out

    # Test filter only with key_201_2009()
    fdesign.print_result(filters.key_201_2009())
    out, _ = capsys.readouterr()
    assert "Filter length   : 201" in out
    assert "Best filter" in out
    assert "> Spacing       : 0.074" in out
    assert "> Shift         : 1.509903313e-14" in out
    assert "> Base min/max  : 6.112528e-04 / 1.635984e+03" in out


def test_ghosh():
    # Assure a DigitalFilter has attributes 'name', 'lhs', and 'rhs'.
    out = fdesign.Ghosh('test', 'lhs', 'rhs')
    assert out.name == 'test'
    assert out.lhs == 'lhs'
    assert out.rhs == 'rhs'


def test_j01():
    # Check with Key201-filter if analytical transform pairs for J0 and J1 are
    # correct
    filt = filters.key_201_2009()
    for j01 in ['1', '2', '3', '4', '5']:
        if j01 == '1':
            r = np.logspace(0, 1.15, 100)
        else:
            r = np.logspace(0, 3, 100)

        k = filt.base/r[:, None]
        tp0 = getattr(fdesign, 'j0_'+j01)(3)
        tp1 = getattr(fdesign, 'j1_'+j01)(3)

        rhs1a = tp0.rhs(r)
        rhs1c = np.dot(tp0.lhs(k), filt.j0)/r

        assert_allclose(rhs1a, rhs1c, rtol=1e-3)

        rhs2a = tp1.rhs(r)
        rhs2c = np.dot(tp1.lhs(k), filt.j1)/r

        assert_allclose(rhs2a, rhs2c, rtol=1e-3)


def test_sincos():
    # Check with Key81-filter if analytical transform pairs for sine and cosine
    # are correct
    filt = filters.key_81_CosSin_2009()
    for sc in ['1', '2', '3']:
        if sc == '1':
            r = np.logspace(0, 0.7, 100)
        else:
            r = np.logspace(0, 1, 100)

        k = filt.base/r[:, None]
        tps = getattr(fdesign, 'sin_'+sc)()
        tpc = getattr(fdesign, 'cos_'+sc)()

        rhs1a = tps.rhs(r)
        rhs1c = np.dot(tps.lhs(k), filt.sin)/r

        assert_allclose(rhs1a, rhs1c, rtol=1e-3)

        rhs2a = tpc.rhs(r)
        rhs2c = np.dot(tpc.lhs(k), filt.cos)/r

        assert_allclose(rhs2a, rhs2c, rtol=1e-3)

    # Check inverse
    for sc in ['1', '2', '3']:
        if sc == '1':
            r = np.logspace(0, 0.7, 100)
        else:
            r = np.logspace(0, 1, 100)

        k = filt.base/r[:, None]
        tps = getattr(fdesign, 'sin_'+sc)()
        tpc = getattr(fdesign, 'cos_'+sc)()
        tps_i = getattr(fdesign, 'sin_'+sc)(inverse=True)
        tpc_i = getattr(fdesign, 'cos_'+sc)(inverse=True)

        rhs1a = tps.rhs(r)
        rhs1c = tps_i.lhs(r)

        assert_allclose(rhs1a, rhs1c)

        rhs2a = tpc.rhs(r)
        rhs2c = tpc_i.lhs(r)

        assert_allclose(rhs2a, rhs2c)


def test_empy_hankel():

    # 1. Simple test to compare ['j0', 'j1'] with 'j0' and 'j1'
    out1 = fdesign.empy_hankel(['j0', 'j1'], 50, 100, [2e14, 1], 1, 0)
    out2 = fdesign.empy_hankel('j0', 50, 100, [2e14, 1], 1, 0)
    out3 = fdesign.empy_hankel('j1', 50, 100, [2e14, 1], 1, 0)
    assert out1[0].name == out2.name
    assert out1[1].name == out3.name

    # 2. Check J0, J1 with analytical, wavenumber
    zsrc = -50
    zrec = 0
    r = np.arange(1, 101)
    f = 100
    model1 = {'res': 100,
              'aniso': 2,
              'epermH': 15,
              'epermV': 30,
              'mpermH': 1,
              'mpermV': 5}
    out4a = fdesign.empy_hankel(['j0', 'j1'], zsrc, zrec, freqtime=f, depth=[],
                                **model1)
    out4b = model.analytical([0, 0, zsrc], [r/np.sqrt(2), r/np.sqrt(2), zrec],
                             freqtime=f, verb=0, **model1)
    out4c = model.analytical([0, 0, zsrc], [r, r*0, zrec], freqtime=f, verb=0,
                             ab=31, **model1)
    out4d, _ = model.dipole_k(src=[0, 0, zsrc],
                              rec=[1/np.sqrt(2), 1/np.sqrt(2), zrec],
                              freq=f, depth=[], wavenumber=1/r, **model1)
    _, out4e = model.dipole_k(src=[0, 0, zsrc], ab=31, rec=[1, 0, zrec],
                              freq=f, depth=[], wavenumber=1/r, **model1)
    assert_allclose(out4a[0].rhs(r), out4b)
    assert_allclose(out4a[1].rhs(r), out4c)
    assert_allclose(out4a[0].lhs(1/r), out4d)
    assert_allclose(out4a[1].lhs(1/r), out4e)

    # 2. Check J2 with dipole, wavenumber
    zsrc = 950
    zrec = 1000
    r = np.arange(1, 101)*20
    f = 0.1
    model2 = {'depth': [0, 1000],
              'res': [2e14, 0.3, 1],
              'aniso': [1, 1, 1.5],
              'epermH': [1, 15, 1],
              'epermV': [1, 1, 30],
              'mpermH': [1, 1, 10],
              'mpermV': [1, 1, 5]}
    out5a = fdesign.empy_hankel('j2', zsrc, zrec, freqtime=f, **model2)
    out5b = model.dipole([0, 0, zsrc], [r/np.sqrt(2), r/np.sqrt(2), zrec],
                         freqtime=f, verb=0, ab=12, **model2)
    out5c, out5d = model.dipole_k(src=[0, 0, zsrc],
                                  rec=[1/np.sqrt(2), 1/np.sqrt(2), zrec],
                                  ab=12, freq=f, wavenumber=1/r, **model2)
    assert_allclose(out5a.rhs(r), out5b)
    assert_allclose(out5a.lhs(1/r)[0], out5c)
    assert_allclose(out5a.lhs(1/r)[1], out5d)


def test_get_min_val(capsys):

    # Some parameters
    fI0 = fdesign.j0_1(5)
    fI1 = fdesign.j1_1(5)
    rdef = (1, 1, 2)
    error = 0.01

    # 1. "Normal" case
    r = np.logspace(0, 2, 10)
    fC = fdesign.j0_1(5)
    fC.rhs = fC.rhs(r)
    out = fdesign._get_min_val((0.05, -1.0), 201, [fI0, ], [fC, ], r, rdef,
                               error, np.real, 'amp', 0, 0, [])
    assert_allclose(out, 2.386523e-05, rtol=1e-5)

    # 2. "Normal" case j0 and j1; J0 is better than J1
    fC1 = fdesign.j1_1(5)
    fC1.rhs = fC1.rhs(r)

    out = fdesign._get_min_val((0.05, -1.0), 201, [fI1, fI0], [fC1, fC], r,
                               rdef, error, np.real, 'amp', 0, 0, [])
    assert_allclose(out, 2.386523e-05, rtol=1e-5)

    # 3. f2
    fC2 = fdesign.empy_hankel('j2', 950, 1000, 1, 1)
    fC2.rhs = fC2.rhs(r)
    out = fdesign._get_min_val((0.05, -1.0), 201, [fI0, fI1], [fC2, ], r, rdef,
                               error, np.real, 'amp', 0, 0, [])
    assert_allclose(out, 6.831394e-08, rtol=1e-5)

    # 4. No solution below error
    out = fdesign._get_min_val((0.05, -10.0), 201, [fI0, ], [fC, ], r, rdef,
                               error, np.real, 'amp', 0, 0, [])
    assert_allclose(out, np.inf)

    # 5. All nan's; with max r
    out = fdesign._get_min_val((0.05, 10.0), 201, [fI0, ], [fC, ], r, rdef,
                               error, np.real, 'r', 0, 0, [])
    assert_allclose(out, np.inf)

    # 6. r too small, with verbosity
    log = {'cnt1': 1, 'cnt2': 9, 'totnr': 10, 'time': default_timer(),
           'warn-r': 0}
    r = np.logspace(0, 1.1, 10)
    fC = fdesign.j0_1(5)
    fC.rhs = fC.rhs(r)
    out, _ = capsys.readouterr()  # empty
    fdesign._get_min_val((0.058, -1.26), 201, [fI0, ], [fC, ], r, rdef, error,
                         np.real, 'amp', 3, 0, log)
    out, _ = capsys.readouterr()
    assert "* WARNING :: all data have error < "+str(error)+";" in out
    assert "brute fct calls : 10" in out


def test_calculate_filter():
    # Test a very small filter (n=3) with know result
    f1 = fdesign._calculate_filter(n=3, spacing=0.77, shift=-0.08,
                                   fI=[fdesign.j0_1(5), ], r_def=(1, 1, 2),
                                   reim=np.real, name='success')
    assert 'success' == f1.name
    assert_allclose(f1.base, [0.42741493, 0.92311635, 1.99371553])
    assert_allclose(f1.j0, [1.08179183, -0.11669046, 0.03220896])
    assert_allclose(f1.factor, 2.1597662537849152)

    # The above means just there was a solution to the inversion. The found
    # solution is most likely a very bad filter, as n=3. So we check it with
    # a transform pair that always works: lhs = 1, rhs = 1/r.
    r = np.logspace(1, 2, 10)
    rhs = np.dot(np.ones((r.size, f1.base.size)), f1.j0)/r
    assert_allclose(1/r, rhs, atol=1e-3)  # Only to 0.1 %, 3 pt filter!

    # Test a very small filter (n=3) with no result (fail)
    f2 = fdesign._calculate_filter(n=4, spacing=0.77, shift=-0.08,
                                   fI=[fdesign.j0_1(5), ], r_def=(0, 0, 0.5),
                                   reim=np.real, name='fail')
    assert 'fail' == f2.name
    assert_allclose(f2.base, [0.1978987, 0.42741493, 0.92311635, 1.99371553]),
    assert_allclose(f2.j0, [0, 0, 0, 0])
    assert_allclose(f2.factor, 2.1597662537849152)


def test_ls2ar():
    # Verify output of ls2ar for different input cases

    # Normal case I-A: tuple/list/array
    tinp = (-1, 1, 20)  # (a) tuple
    tout = fdesign._ls2ar(tinp, 'Normal')
    assert_allclose(np.arange(*tout), np.linspace(*tinp))

    # Weird case I-B: tuple/list/array
    tinp = (-10, -1, 0)  # (a) tuple
    tout = fdesign._ls2ar(tinp, 'Weird')
    assert_allclose(np.arange(*tout), np.array([-10]))

    # Normal case II-A: One value, tuple/list/array
    tinp = [np.e, -np.pi, 1]  # (b) list
    tout = fdesign._ls2ar(tinp, 'ListFloat')
    assert_allclose(np.arange(*tout), np.array([np.e]))

    # Normal case II-B: float
    tout = fdesign._ls2ar(np.pi, 'Float')  # (c) float
    assert_allclose(np.arange(*tout), np.array([np.pi]))

    # Error
    with pytest.raises(ValueError):
        fdesign._ls2ar(np.array([1, 2, 3, 4]), 'TooMany')  # (d) array


def test_print_count(capsys):
    # first count
    out = fdesign._print_count({'cnt1': -1, 'cnt2': -1, 'totnr': 10,
                                'time': default_timer(), 'warn-r': 0})
    out, _ = capsys.readouterr()
    assert out == ""

    # fmin counts
    out = fdesign._print_count({'cnt1': 10, 'cnt2': 100, 'totnr': 10,
                                'time': default_timer(), 'warn-r': 0})
    out, _ = capsys.readouterr()
    assert "   fmin  fct calls : 91" in out

    # cnt2 == totnr
    out = fdesign._print_count({'cnt1': 10, 'cnt2': 199, 'totnr': 200,
                                'time': default_timer(), 'warn-r': 0})
    out, _ = capsys.readouterr()
    assert "   brute fct calls : 200" in out

    # cp < 1
    out = fdesign._print_count({'cnt1': 0, 'cnt2': 9, 'totnr': 200,
                                'time': default_timer(), 'warn-r': 0})
    out, _ = capsys.readouterr()
    assert "   brute fct calls : 10/200 (5 %); est: 0:00:" in out

    # cp > cnt1
    out = fdesign._print_count({'cnt1': 9, 'cnt2': 19, 'totnr': 200,
                                'time': default_timer(), 'warn-r': 0})
    out, _ = capsys.readouterr()
    assert "   brute fct calls : 20/200 (10 %); est: 0:00:" in out

    # nothing
    out = fdesign._print_count({'cnt1': 10, 'cnt2': 19, 'totnr': 200,
                                'time': default_timer(), 'warn-r': 0})
    out, _ = capsys.readouterr()
    assert out == ""
