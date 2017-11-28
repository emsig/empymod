import numpy as np
from numpy.testing import assert_allclose

from empymod import filters


def test_digitalfilter():                                    # 1. DigitalFilter
    # Assure a DigitalFilter has attribute 'name'.
    out = filters.DigitalFilter('test')
    assert out.name == 'test'


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
