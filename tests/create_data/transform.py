r"""Create data for test_transform."""
import numpy as np
from scipy.constants import mu_0
from scipy import special
from scipy.interpolate import InterpolatedUnivariateSpline as iuSpline

from empymod import utils, kernel


# # A -- Analytical Solutions # #
# These analytical solutions are difficult for the Fourier-transform, as source
# and receiver are at the air interface (z=0). Any other model, where source
# and receiver are not at the air interface, will be much easier to transform,
# and ftarg can be relaxed by still getting a better precision!

def test_freq(res, off, f):
    r"""Frequency domain analytical half-space solution.
    - Source at x = y = z = 0 m
    - Receiver at y = z = 0 m; x = off
    - Resistivity of halfspace res
    - Frequencies f
    """
    gamma = np.sqrt(mu_0*2j*np.pi*f*off**2/res)
    return res/(2*np.pi*off**3)*(1 + (1 + gamma)*np.exp(-gamma))


def test_time(res, off, t, signal):
    r"""Time domain analytical half-space solution.
    - Source at x = y = z = 0 m
    - Receiver at y = z = 0 m; x = off
    - Resistivity of halfspace res
    - Times t, t > 0 s
    - Impulse response if signal = 0
    - Switch-on response if signal = 1
    """
    tau = np.sqrt(mu_0*off**2/(res*t))
    fact1 = res/(2*np.pi*off**3)
    fact2 = tau/np.sqrt(np.pi)
    if signal == 0:
        return fact1*tau**3/(4*t*np.sqrt(np.pi))*np.exp(-tau**2/4)
    else:
        resp = fact1*(2 - special.erf(tau/2) + fact2*np.exp(-tau**2/4))
        if signal < 0:
            DC = test_time(res, off, 1000000, 1)
            resp = DC-resp
        return resp


# Time-domain solution
res = 20  # Ohm.m
off = 4000  # m
t = np.logspace(-1.5, 1, 20)  # s
tEM0 = test_time(res, off, t, 0)
tEM1 = test_time(res, off, t, 1)
tEM2 = test_time(res, off, t, -1)


# # B -- Fourier FFTLog # #
# Signal = 0
_, f, _, ftarg = utils.check_time(t, 0, 'fftlog', ['', [-3, 2]], 0)
fEM = test_freq(res, off, f)
fourier_fftlog0 = {'fEM': fEM, 'f': f, 'ftarg': ftarg}
# Signal = 1
_, f, _, ftarg = utils.check_time(t, 1, 'fftlog', [30, [-3, 2], -.5], 0)
fEM = test_freq(res, off, f)
fourier_fftlog1 = {'fEM': fEM, 'f': f, 'ftarg': ftarg}
# Signal = -1
_, f, _, ftarg = utils.check_time(t, -1, 'fftlog', [30, [-5, 2], .1], 0)
fEM = test_freq(res, off, f)
fourier_fftlog2 = {'fEM': fEM, 'f': f, 'ftarg': ftarg}

# # C -- Fourier DLF # #
# Signal = 0
_, f, _, ftarg = utils.check_time(t, 0, 'cos', {'pts_per_dec': 0}, 0)
fEM = test_freq(res, off, f)
fourier_dlf0 = {'fEM': fEM, 'f': f, 'ftarg': ftarg}
# Signal = 1
_, f, _, ftarg = utils.check_time(t, 1, 'sin', ['key_201_CosSin_2012', -1], 0)
fEM = test_freq(res, off, f)
fourier_dlf1 = {'fEM': fEM, 'f': f, 'ftarg': ftarg}
# Signal = -1
_, f, _, ftarg = utils.check_time(t, -1, 'sin', ['key_201_CosSin_2012', 20], 0)
fEM = test_freq(res, off, f)
fourier_dlf2 = {'fEM': fEM, 'f': f, 'ftarg': ftarg}

# # D -- Fourier QWE # #
# Signal = 0
_, f, _, ftarg = utils.check_time(t, 0, 'qwe', None, 0)
fEM = test_freq(res, off, f)
fourier_qwe0 = {'fEM': fEM, 'f': f, 'ftarg': ftarg}
# Signal = 1
_, f, _, ftarg = utils.check_time(t, 1, 'qwe', [1e-6, '', 41, 300, '', '',
                                                1e-4, 1e4, 1000], 0)
fEM = test_freq(res, off, f)
fourier_qwe1 = {'fEM': fEM, 'f': f, 'ftarg': ftarg}
# Signal = -1
_, f, _, ftarg = utils.check_time(t, -1, 'qwe', [1e-6, '', 41, 300, '', '',
                                                 1e-5, 1e5, 1000], 0)
fEM = test_freq(res, off, f)
fourier_qwe2 = {'fEM': fEM, 'f': f, 'ftarg': ftarg}

# # E -- Fourier FFT # #
# Signal = 0
_, f, _, ftarg = utils.check_time(t, 0, 'fft', [0.0005, 2**20, '', 10], 0)
fEM = test_freq(res, off, f)
fourier_fft0 = {'fEM': fEM, 'f': f, 'ftarg': ftarg}

# # F -- QWE - Fourier QWE # #
nquad = fourier_qwe0['ftarg']['nquad']
maxint = fourier_qwe0['ftarg']['maxint']
fEM = fourier_qwe0['fEM']
freq = fourier_qwe0['f']
# The following is a condensed version of transform.fourier_qwe, without
# doqwe-part
xint = np.concatenate((np.array([1e-20]), np.arange(1, maxint+1)*np.pi))
intervals = xint/t[:, None]
g_x, g_w = special.p_roots(nquad)
dx = np.repeat(np.diff(xint)/2, nquad)
Bx = dx*(np.tile(g_x, maxint) + 1) + np.repeat(xint[:-1], nquad)
SS = np.sin(Bx)*np.tile(g_w, maxint)
tEM_iint = iuSpline(np.log(2*np.pi*freq), fEM.imag)
sEM = tEM_iint(np.log(Bx/t[:, None]))*SS
fourier_qwe0['sEM'] = sEM
fourier_qwe0['intervals'] = intervals

# # G -- QWE - HANKEL QWE # #
# Model
model = utils.check_model([], 10, 2, 2, 5, 1, 10, True, 0)
depth, res, aniso, epermH, epermV, mpermH, mpermV, isfullspace = model
frequency = utils.check_frequency(1, res, aniso, epermH, epermV, mpermH,
                                  mpermV, 0)
freq, etaH, etaV, zetaH, zetaV = frequency
src, nsrc = utils.check_dipole([0, 0, 0], 'src', 0)
ab, msrc, mrec = utils.check_ab(11, 0)
ht, htarg = utils.check_hankel('qwe', {'pts_per_dec': 80}, 0)
rec = [np.arange(1, 11)*500, np.zeros(10), 300]
rec, nrec = utils.check_dipole(rec, 'rec', 0)
off, angle = utils.get_off_ang(src, rec, nsrc, nrec, 0)
lsrc, zsrc = utils.get_layer_nr(src, depth)
lrec, zrec = utils.get_layer_nr(rec, depth)
# Frequency-domain result
freqres = kernel.fullspace(off, angle, zsrc, zrec, etaH, etaV, zetaH, zetaV,
                           ab, msrc, mrec)
# The following is a condensed version of transform.hankel_qwe
etaH = etaH[0, :]
etaV = etaV[0, :]
zetaH = zetaH[0, :]
zetaV = zetaV[0, :]
rtol = htarg['rtol']
atol = htarg['atol']
nquad = htarg['nquad']
maxint = htarg['maxint']
pts_per_dec = htarg['pts_per_dec']
diff_quad = htarg['diff_quad']
a = htarg['a']
b = htarg['b']
limit = htarg['limit']
g_x, g_w = special.p_roots(nquad)
b_zero = np.pi*np.arange(1.25, maxint+1)
for i in range(10):
    b_x0 = special.j1(b_zero)
    b_x1 = special.jv(2, b_zero)
    b_h = -b_x0/(b_x0/b_zero - b_x1)
    b_zero += b_h
    if all(np.abs(b_h) < 8*np.finfo(float).eps*b_zero):
        break
xint = np.concatenate((np.array([1e-20]), b_zero))
dx = np.repeat(np.diff(xint)/2, nquad)
Bx = dx*(np.tile(g_x, maxint) + 1) + np.repeat(xint[:-1], nquad)
BJ0 = special.j0(Bx)*np.tile(g_w, maxint)
BJ1 = special.j1(Bx)*np.tile(g_w, maxint)
intervals = xint/off[:, None]
lambd = Bx/off[:, None]
ang_fact = kernel.angle_factor(angle, ab, msrc, mrec)
# 1 Spline version
start = np.log(lambd.min())
stop = np.log(lambd.max())
ilambd = np.logspace(start, stop, int((stop-start)*pts_per_dec + 1), base=10.0)
PJ0, PJ1, PJ0b = kernel.wavenumber(zsrc, zrec, lsrc, lrec, depth,
                                   etaH[None, :], etaV[None, :],
                                   zetaH[None, :], zetaV[None, :],
                                   np.atleast_2d(ilambd), ab, False,
                                   msrc, mrec)
si_PJ0r = iuSpline(np.log(ilambd), PJ0.real)
si_PJ0i = iuSpline(np.log(ilambd), PJ0.imag)
si_PJ1r = iuSpline(np.log(ilambd), PJ1.real)
si_PJ1i = iuSpline(np.log(ilambd), PJ1.imag)
si_PJ0br = iuSpline(np.log(ilambd), PJ0b.real)
si_PJ0bi = iuSpline(np.log(ilambd), PJ0b.imag)
sPJ0 = si_PJ0r(np.log(lambd))+1j*si_PJ0i(np.log(lambd))
sPJ1 = si_PJ1r(np.log(lambd))+1j*si_PJ1i(np.log(lambd))
sPJ0b = si_PJ0br(np.log(lambd))+1j*si_PJ0bi(np.log(lambd))
sEM = np.sum(np.reshape(sPJ1*BJ1, (off.size, nquad, -1), order='F'), 1)
if ab in [11, 12, 21, 22, 14, 24, 15, 25]:  # Because of J2
    # J2(kr) = 2/(kr)*J1(kr) - J0(kr)
    sEM /= np.atleast_1d(off[:, np.newaxis])
sEM += np.sum(np.reshape(sPJ0b*BJ0, (off.size, nquad, -1), order='F'), 1)
sEM *= ang_fact[:, np.newaxis]
sEM += np.sum(np.reshape(sPJ0*BJ0, (off.size, nquad, -1), order='F'), 1)

# Additinol stuff needed for non-spline version
nsinp = {'zsrc': zsrc, 'zrec': zrec, 'lsrc': lsrc, 'lrec': lrec, 'depth':
         depth, 'etaH': etaH[None, :], 'etaV': etaV[None, :], 'zetaH':
         zetaH[None, :], 'zetaV': zetaV[None, :], 'ab': ab, 'xdirect': False,
         'msrc': msrc, 'mrec': mrec}

hankel_qwe = {
        'rtol': rtol, 'atol': atol, 'maxint': maxint, 'getkernel': sEM,
        'intervals': intervals, 'lambd': lambd, 'off': off,
        'ang_fact': ang_fact, 'nsinp': nsinp, 'nquad': nquad, 'BJ0': BJ0,
        'BJ1': BJ1, 'ab': ab, 'freqres': np.squeeze(freqres),
        'diff_quad': diff_quad}

# # H -- Hankel QUAD # #
# Model
model = utils.check_model([], 10, 2, 2, 5, 1, 10, True, 0)
depth, res, aniso, epermH, epermV, mpermH, mpermV, isfullspace = model
frequency = utils.check_frequency(1, res, aniso, epermH, epermV, mpermH,
                                  mpermV, 0)
freq, etaH, etaV, zetaH, zetaV = frequency
src, nsrc = utils.check_dipole([0, 0, 0], 'src', 0)
ab, msrc, mrec = utils.check_ab(11, 0)
ht, htarg = utils.check_hankel('quad', None, 0)
rec = [5000, 0, 300]
rec, nrec = utils.check_dipole(rec, 'rec', 0)
off, angle = utils.get_off_ang(src, rec, nsrc, nrec, 0)
lsrc, zsrc = utils.get_layer_nr(src, depth)
lrec, zrec = utils.get_layer_nr(rec, depth)
# Frequency-domain result
freqres = kernel.fullspace(off, angle, zsrc, zrec, etaH, etaV, zetaH, zetaV,
                           ab, msrc, mrec)
# The following is a condensed version of transform.hankel_quad
rtol = htarg['rtol']
atol = htarg['atol']
limit = htarg['limit']
a = htarg['a']
b = htarg['b']
pts_per_dec = htarg['pts_per_dec']
la = np.log(a)
lb = np.log(b)
ilambd = np.logspace(la, lb, int((lb-la)*pts_per_dec + 1), base=np.e)
PJ0, PJ1, PJ0b = kernel.wavenumber(zsrc, zrec, lsrc, lrec, depth,
                                   etaH, etaV, zetaH, zetaV,
                                   np.atleast_2d(ilambd), ab, False, msrc,
                                   mrec)
sPJ0r = iuSpline(np.log(ilambd), PJ0.real)
sPJ0i = iuSpline(np.log(ilambd), PJ0.imag)
sPJ1r = iuSpline(np.log(ilambd), PJ1.real)
sPJ1i = iuSpline(np.log(ilambd), PJ1.imag)
sPJ0br = iuSpline(np.log(ilambd), PJ0b.real)
sPJ0bi = iuSpline(np.log(ilambd), PJ0b.imag)
ang_fact = kernel.angle_factor(angle, ab, msrc, mrec)
iinp = {'a': a, 'b': b, 'epsabs': atol, 'epsrel': rtol, 'limit': limit}
quad = {'inp': {'sPJ0r': sPJ0r, 'sPJ0i': sPJ0i, 'sPJ1r': sPJ1r, 'sPJ1i': sPJ1i,
                'sPJ0br': sPJ0br, 'sPJ0bi': sPJ0bi, 'ab': ab, 'off': off,
                'ang_fact': ang_fact, 'iinp': iinp},
        'res': np.squeeze(freqres)}

# # I -- Store data # #
np.savez_compressed(
        '../data/transform.npz', t=t, tEM0=tEM0, tEM1=tEM1, tEM2=tEM2,
        fourier_fftlog0=fourier_fftlog0, fourier_fftlog1=fourier_fftlog1,
        fourier_fftlog2=fourier_fftlog2,
        fourier_dlf0=fourier_dlf0, fourier_dlf1=fourier_dlf1,
        fourier_dlf2=fourier_dlf2,
        fourier_qwe0=fourier_qwe0, fourier_qwe1=fourier_qwe1,
        fourier_qwe2=fourier_qwe2,
        # fft1/2 are dummies
        fourier_fft0=fourier_fft0, fourier_fft1=fourier_fft0,
        fourier_fft2=fourier_fft0,
        hankel_qwe=hankel_qwe, quad=quad)
