import pytest
import numpy as np
from numpy.testing import assert_allclose

from empymod.model import dipole, frequency, time

# model. Status: 0/13
# These are kind of macro-tests, as they check the final results.

# Set of models
# Set of surveys
# All ab's
# Arbitrary angles
# All transforms
# All optimizations
# Source/Receiver scaling

class SingleSurvEx:
    def test_single(self):
        src = [0, 0, 500]
        rec = [5000, 1000, 600]
        depth = 0
        signal = 1
        res = [1, 10]
        freq = [0.01, 1, 100]
        ab = 12
        aniso = [1, 3]
        epermH = [1, 80]
        epermV = [1, 40]
        mpermH = [1, 5]
        mpermV = [1, 10]
        xdirect = True
        ht = 'fht'
        htarg = ['key_401_2009', None]
        ft = 'fftlog'
        ftarg = [10, [-2, 1], -.5]
        opt = 'spline'
        loop = 'freq'
        verb = 0
        self.do(src, rec, depth, signal, res, freq, ab, aniso, epermH, epermV,
                mpermH, mpermV, xdirect, ht, htarg, ft, ftarg, opt, loop, verb)

    def test_double(self):
        src = [0, 0, 500]
        rec = [np.arange(1,11)*500, np.ones(10)*1000, 600]
        depth = 0
        signal = 1
        res = [1, 10]
        freq = [0.01, 1, 100]
        ab = 12
        aniso = [1, 3]
        epermH = [1, 80]
        epermV = [1, 40]
        mpermH = [1, 5]
        mpermV = [1, 10]
        xdirect = True
        ht = 'fht'
        htarg = ['key_401_2009', None]
        ft = 'fftlog'
        ftarg = [10, [-2, 1], -.5]
        opt = 'spline'
        loop = 'freq'
        verb = 0
        self.do(src, rec, depth, signal, res, freq, ab, aniso, epermH, epermV,
                mpermH, mpermV, xdirect, ht, htarg, ft, ftarg, opt, loop, verb)

# 0. __init__ (Ensure main functions are importable from empymod)

# 1. bipole

# 1.1. Comparison to analytical fullspace solution

# 1.2. Comparison to analytical halfspace solution

# 1.3. Comparison to EMmod

# 1.4. Comparison to DIPOLE1D

# 1.5. Comparison to Green3D

# 2. dipole (Ensure it is equivalent to bipole)

# 3. gpr (Check it remains as in paper)

# 4. wavenumber (Finish wavenumber properly; write checks)

# 5. frequency: As this is a shortcut, just run one test to ensure it is
#               equivalent to dipole with signal=None.
class TestFrequency(SingleSurvEx):
    def do(self, src, rec, depth, signal, res, freq, ab, aniso, epermH, epermV,
           mpermH, mpermV, xdirect, ht, htarg, ft, ftarg, opt, loop, verb):
        f_res = frequency(src, rec, depth, res, freq, ab, aniso, epermH,
                          epermV, mpermH, mpermV, xdirect, ht, htarg, opt,
                          loop, verb)
        d_res = dipole(src, rec, depth, res, freq, None, ab, aniso, epermH,
                       epermV, mpermH, mpermV, xdirect, ht, htarg, opt, loop,
                       verb)
        assert_allclose(f_res, d_res, rtol=1e-7, atol=1e-14)

# 6. time: As this is a shortcut, just run one test to ensure it is equivalent
#          to dipole with signal!=None.
class TestTime(SingleSurvEx):
    def do(self, src, rec, depth, signal, res, freq, ab, aniso, epermH, epermV,
           mpermH, mpermV, xdirect, ht, htarg, ft, ftarg, opt, loop, verb):

        t_res = time(src, rec, depth, res, freq, ab, signal, aniso, epermH,
                     epermV, mpermH, mpermV, xdirect, ht, htarg, ft, ftarg,
                     opt, loop, verb)
        d_res = dipole(src, rec, depth, res, freq, signal, ab, aniso, epermH,
                       epermV, mpermH, mpermV, xdirect, ht, htarg, ft, ftarg,
                       opt, loop, verb)
        assert_allclose(t_res, d_res, rtol=1e-7, atol=1e-14)


# 7. fem

# 8. tem
