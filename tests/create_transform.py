"""Create data for test_transform."""
import numpy as np
from scipy.special import erf
from scipy.constants import mu_0

from empymod.utils import check_time


# # A -- Analytical Solutions # #
# These analytical solutions are difficult for the Fourier-transform, as source
# and receiver are at the air interface (z=0). Any other model, where source
# and receiver are not at the air interface, will be much easier to transform,
# and ftarg can be relaxed by still getting a better precision!

def test_freq(res, off, f):
    """Frequency domain analytical half-space solution.
    - Source at x = y = z = 0 m
    - Receiver at y = z = 0 m; x = off
    - Resistivity of halfspace res
    - Frequencies f
    """
    gamma = np.sqrt(mu_0*2j*np.pi*f*off**2/res)
    return res/(2*np.pi*off**3)*(1 + (1 + gamma)*np.exp(-gamma))


def test_time(res, off, t, signal):
    """Time domain analytical half-space solution.
    - Source at x = y = z = 0 m
    - Receiver at y = z = 0 m; x = off
    - Resistivity of halfspace res
    - Times t, t > 0 s
    - Impulse response if signal = 0
    - Switch-on response if signal = 1
    """
    tau = np.sqrt(mu_0*off**2/(res*t))
    fact = res/(2*np.pi*off**3)
    if signal == 1:
        return fact*(2 - erf(tau/2) + tau/np.sqrt(np.pi)*np.exp(-tau**2/4))
    elif signal == 0:
        return fact*tau**3/(4*t*np.sqrt(np.pi))*np.exp(-tau**2/4)

# Time-domain solution
res = 20  # Ohm.m
off = 4000  # m
t = np.logspace(-1.5, 1, 20)  # s
tEM0 = test_time(res, off, t, 0)
tEM1 = test_time(res, off, t, 1)


# # B -- FFTLog # #
# Signal = 0
_, f, _, ftarg = check_time(t, 0, 'fftlog', ['', [-3, 2]], 0)
fEM = test_freq(res, off, f)
fftlog0 = {'fEM': fEM, 'f': f, 'ftarg': ftarg}
# Signal = 1
_, f, _, ftarg = check_time(t, 1, 'fftlog', [30, [-3, 2], -.5], 0)
fEM = test_freq(res, off, f)
fftlog1 = {'fEM': fEM, 'f': f, 'ftarg': ftarg}

# # C -- FFT # #
# Signal = 0
_, f, _, ftarg = check_time(t, 0, 'cos', None, 0)
fEM = test_freq(res, off, f)
fft0 = {'fEM': fEM, 'f': f, 'ftarg': ftarg}
# Signal = 1
_, f, _, ftarg = check_time(t, 1, 'sin', ['key_201_CosSin_2012', 20], 0)
fEM = test_freq(res, off, f)
fft1 = {'fEM': fEM, 'f': f, 'ftarg': ftarg}

# # D -- FQWE # #
# Signal = 0
_, f, _, ftarg = check_time(t, 0, 'qwe', None, 0)
fEM = test_freq(res, off, f)
fqwe0 = {'fEM': fEM, 'f': f, 'ftarg': ftarg}
# Signal = 1
_, f, _, ftarg = check_time(t, 1, 'qwe', [1e-6, '', 41, 300], 0)
fEM = test_freq(res, off, f)
fqwe1 = {'fEM': fEM, 'f': f, 'ftarg': ftarg}

# # F -- Store data # #
np.savez_compressed('data_transform.npz',
                    t=t, tEM0=tEM0, tEM1=tEM1,
                    fftlog0=fftlog0, fftlog1=fftlog1,
                    fft0=fft0, fft1=fft1,
                    fqwe0=fqwe0, fqwe1=fqwe1)
