import os
import sys
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


@pytest.mark.skipif(sys.version_info < (3, 6),
                    reason="tmpdir seems to fail for Python<3.6.")
def test_storeandsave(tmpdir):                                  # 1.b Save/Load
    # Store a filter
    inpfilt = filters.wer_201_2018()
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


def test_fhtfilters():                                         # 2. FHT filters
    # Check that all FHT filters
    #   (a) exist,
    #   (b) base, j0, and j1 have right number of values
    #       (nothing got accidently deleted), and
    #   (c) factor is correct.
    allfilt = ['kong_61_2007', 'kong_241_2007', 'key_101_2009', 'key_201_2009',
               'key_401_2009', 'anderson_801_1982', 'key_51_2012',
               'key_101_2012', 'key_201_2012', 'wer_201_2018']
    for filt in allfilt:
        fhtfilt = getattr(filters, filt)()
        nr = int(filt.split('_')[1])
        fact = np.around(np.average(fhtfilt.base[1:]/fhtfilt.base[:-1]), 15)
        assert len(fhtfilt.base) == nr
        assert len(fhtfilt.j0) == nr
        assert len(fhtfilt.j1) == nr
        assert_allclose(fhtfilt.factor, fact)


def test_co_sinefilters():                                 # 3. Co/Sine filters
    # Check that all Co/Sine filters
    #   (a) exist,
    #   (b) base, j0, and j1 have right number of values
    #       (nothing got accidently deleted), and
    #   (c) factor is correct.
    allfilt = ['key_81_CosSin_2009', 'key_241_CosSin_2009',
               'key_601_CosSin_2009', 'key_101_CosSin_2012',
               'key_201_CosSin_2012']
    for filt in allfilt:
        fhtfilt = getattr(filters, filt)()
        nr = int(filt.split('_')[1])
        fact = np.around(np.average(fhtfilt.base[1:]/fhtfilt.base[:-1]), 15)
        assert len(fhtfilt.base) == nr
        assert len(fhtfilt.cos) == nr
        assert len(fhtfilt.sin) == nr
        assert_allclose(fhtfilt.factor, fact)
