import pytest
# import numpy as np
from numpy.testing import assert_allclose

# Import main ones from empymod directly to check
from empymod import bipole, dipole, frequency, time
# Rest from model
# from empymod.model import ...

# model. Status: 3/13
# These are kind of macro-tests, as they check the final results.

# Different Sets of Sources
dipSources = [
        [0, 0, 400],
        [100, -100, 400]
        ]

# Different Sets of Receivers
dipReceivers = [
        [1000, 0, 500],
        [1000, 1000, 1000]
        ]

# Different Sets of Models
simpleModels = [
        {'depth': 0, 'res': [1e12, 10], 'aniso': [1, 1]},
        {'depth': [0, 500], 'res': [1e12, 0.3, 10], 'aniso': [1, 1, 2]}
        ]


# Create different types of surveys
class SimpleDipoleSurvey:
    @pytest.mark.parametrize("src", dipSources)
    @pytest.mark.parametrize("rec", dipReceivers)
    @pytest.mark.parametrize("model", simpleModels)
    def test_single(self, src, rec, model):
        self.do(src, rec, model)


# 1. bipole  => Main and most important checks/comparisons

# 1.1. Comparison to analytical fullspace solution

# 1.2. Comparison to analytical halfspace solution

# 1.3. Comparison to EMmod

# 1.4. Comparison to DIPOLE1D

# 1.5. Comparison to Green3D

# 2. dipole (Ensure it is equivalent to bipole)
class TestDipole(SimpleDipoleSurvey):                               # 2. dipole
    # As this is a shortcut, just run one test to ensure
    # it is equivalent to bipole.
    def do(self, src, rec, model):
        f = 0.01
        # v  dipole : ab = 26
        # \> bipole : src-dip = 90, rec-azimuth=90, msrc=True
        dip_res = dipole(src, rec, freqtime=f, ab=26, **model)
        bip_res = bipole([src[0], src[1], src[2], 0, 90],
                         [rec[0], rec[1], rec[2], 90, 0],
                         msrc=True, freqtime=f, **model)
        assert_allclose(dip_res, bip_res)

# 3. gpr (Check it remains as in paper)

# 4. wavenumber (Finish wavenumber properly; write checks)


class TestFrequency(SimpleDipoleSurvey):                         # 5. frequency
    # As this is a shortcut, just run one test to ensure
    # it is equivalent to dipole with signal=None.
    def do(self, src, rec, model):
        f = 1
        ab = 45
        f_res = frequency(src, rec, freq=f, ab=ab, **model)
        d_res = dipole(src, rec, freqtime=f, ab=ab, **model)
        assert_allclose(f_res, d_res)


class TestTime(SimpleDipoleSurvey):                                   # 6. time
    # As this is a shortcut, just run one test to ensure
    # it is equivalent to dipole with signal!=None.
    def do(self, src, rec, model):
        t = 10
        ab = 51
        signal = -1
        ft = 'fftlog'
        t_res = time(src, rec, time=t, signal=signal, ab=ab, ft=ft, **model)
        d_res = dipole(src, rec, freqtime=t, signal=signal, ab=ab, ft=ft,
                       **model)
        assert_allclose(t_res, d_res)


# 7. fem

# 8. tem
