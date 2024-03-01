import os
import pytest
import numpy as np
from numpy.testing import assert_allclose

from empymod import filters


def test_digitalfilter():                                   # 1.a DigitalFilter
    # Assure a DigitalFilter has attribute 'name'.
    out1 = filters.DigitalFilter('test')
    out2 = filters.DigitalFilter('test', 'savenametest')
    out3 = filters.DigitalFilter('test', filter_coeff=['abc', ])
    assert out1.name == 'test'
    assert out1.savename == out1.name
    assert out1.name == out2.name
    assert out1.filter_coeff == ['j0', 'j1', 'sin', 'cos']
    assert out2.savename == 'savenametest'
    assert out3.filter_coeff == ['j0', 'j1', 'sin', 'cos', 'abc']


def test_storeandsave(tmpdir):                                  # 1.b Save/Load
    # Store a filter
    inpfilt = filters.Hankel().wer_201_2018
    inpfilt.savename = 'savetest'
    inpfilt.tofile(tmpdir)
    assert len(tmpdir.listdir()) == 3
    assert os.path.isfile(os.path.join(tmpdir, 'savetest_base.txt')) is True
    assert os.path.isfile(os.path.join(tmpdir, 'savetest_j0.txt')) is True
    assert os.path.isfile(os.path.join(tmpdir, 'savetest_j1.txt')) is True

    # Load a filter
    outfilt = filters.DigitalFilter('savetest')
    outfilt.fromfile(tmpdir)
    assert_allclose(outfilt.base, inpfilt.base)
    assert_allclose(outfilt.j0, inpfilt.j0)
    assert_allclose(outfilt.j1, inpfilt.j1)
    assert_allclose(outfilt.factor, inpfilt.factor)


def test_hankel_dlf():                                      # 2. Hankel filters
    # Check that all Hankel filters
    #   (a) exist,
    #   (b) base, j0, and j1 have right number of values
    #       (nothing got accidently deleted), and
    #   (c) factor is correct.
    allfilt = ['kong_61_2007b', 'kong_241_2007', 'key_101_2009',
               'key_201_2009', 'key_401_2009', 'anderson_801_1982',
               'key_51_2012', 'key_101_2012', 'key_201_2012', 'wer_201_2018']
    H = filters.Hankel()
    for filt in allfilt:
        dlf = getattr(H, filt)
        nr = int(filt.split('_')[1])
        fact = np.around(np.average(dlf.base[1:]/dlf.base[:-1]), 15)
        assert len(dlf.base) == nr
        assert len(dlf.j0) == nr
        assert len(dlf.j1) == nr
        assert_allclose(dlf.factor, fact)

        # Test deprecated way
        with pytest.warns(FutureWarning, match='in v3.0; use'):
            if filt == 'kong_61_2007b':
                filt = filt[:-1]
            dlf0 = getattr(filters, filt)()
            assert dlf == dlf0


def test_fourier_dlf():                                    # 3. Fourier filters
    # Check that all Fourier filters
    #   (a) exist,
    #   (b) base, j0, and j1 have right number of values
    #       (nothing got accidently deleted), and
    #   (c) factor is correct.
    allfilt = ['key_81_2009', 'key_241_2009', 'key_601_2009', 'key_101_2012',
               'key_201_2012']
    F = filters.Fourier()
    for filt in allfilt:
        dlf = getattr(F, filt)
        nr = int(filt.split('_')[1])
        fact = np.around(np.average(dlf.base[1:]/dlf.base[:-1]), 15)
        assert len(dlf.base) == nr
        assert len(dlf.cos) == nr
        assert len(dlf.sin) == nr
        assert_allclose(dlf.factor, fact)

        # Test deprecated way
        with pytest.warns(FutureWarning, match='in v3.0; use'):
            tmp = filt.split("_")
            filt = tmp[0] + "_" + tmp[1] + "_CosSin_" + tmp[2]
            dlf0 = getattr(filters, filt)()
            assert dlf == dlf0


def test_all_dir():
    assert set(filters.__all__) == set(dir(filters))
