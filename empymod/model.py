"""

:mod:`model` -- Model EM-responses
==================================

EM-modelling routines. So far implemented:
    - `frequency`:  Calculate the electromagnetic field for various frequencies
                    and offsets.
    - `time`:       Calculate the electromagnetic field for various times
                    and offsets.

The above routines make use of the two core routines:
    - `fem`:        Calculate wavenumber-domain electromagnetic field and carry
                    out the Hankel transform to the frequency domain.
    - `tem`:        Calculate `fem` and carry out Fourier transform to time
                    domain.

Two more routines are more kind of examples and cannot be regarded stable:
    - `gpr`:        Calculate the Ground-Penetrating Radar (GPR) response.
    - `wavenumber`: Calculate the electromagnetic wavenumber-domain solution.


"""
# Copyright 2016 Dieter Werthmüller
#
# This file is part of `empymod`.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License.  You may obtain a copy
# of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  See the
# License for the specific language governing permissions and limitations under
# the License.


import numpy as np

from . import kernel, transform, utils


__all__ = ['frequency', 'time', 'gpr', 'wavenumber', 'fem', 'tem']


def frequency(src, rec, depth, res, freq, ab=11, aniso=None, epermH=None,
              epermV=None, mpermH=None, mpermV=None, xdirect=True, ht='fht',
              htarg=None, opt=None, loop=None, verb=1):
    """Return the electromagnetic frequency-domain field.

    Calculate the electromagnetic field for various frequencies and offsets
    (one src-rec-configuration).

    Parameters
    ----------
    src : list of floats
        Source coordinates (m): [src-x, src-y, src-z]. All single values (no
        arrays)

    rec : list of floats or arrays
        Receiver coordinates (m): [rec-x, rec-y, rec-z].
        The x- and y-coordinates can be arrays, z is a single value.
        The x- and y-coordinates must have the same dimension.

    depth : list
        Absolute layer boundaries z (m); #depth = #res - 1
        (excluding +/- infinity).

    res : array_like
        Horizontal resistivities rho_h (Ohm.m); #res = #depth + 1.

    freq : array_like
        Frequencies f (Hz)

    ab : int, optional
        Source-receiver configuration, defaults to 11.

        +---------------+-------+------+------+------+------+------+------+
        |                       | electric  source   | magnetic source    |
        +===============+=======+======+======+======+======+======+======+
        |                       | **x**| **y**| **z**| **x**| **y**| **z**|
        +---------------+-------+------+------+------+------+------+------+
        |               | **x** |  11  |  12  |  13  |  14  |  15  |  16  |
        + **electric**  +-------+------+------+------+------+------+------+
        |               | **y** |  21  |  22  |  23  |  24  |  25  |  26  |
        + **receiver**  +-------+------+------+------+------+------+------+
        |               | **z** |  31  |  32  |  33  |  34  |  35  |  36  |
        +---------------+-------+------+------+------+------+------+------+
        |               | **x** |  41  |  42  |  43  |  44  |  45  |  46  |
        + **magnetic**  +-------+------+------+------+------+------+------+
        |               | **y** |  51  |  52  |  53  |  54  |  55  |  56  |
        + **receiver**  +-------+------+------+------+------+------+------+
        |               | **z** |  61  |  62  |  63  |  64  |  65  |  66  |
        +---------------+-------+------+------+------+------+------+------+

    aniso : array_like, optional
        Anisotropies lambda = sqrt(rho_v/rho_h) (-); #aniso = #res.
        Defaults to ones.

    epermH : array_like, optional
        Horizontal electric permittivities epsilon_h (-); #epermH = #res.
        Defaults to ones.

    epermV : array_like, optional
        Vertical electric permittivities epsilon_v (-); #epermV = #res.
        Defaults to ones.

    mpermH : array_like, optional
        Horizontal magnetic permeabilities mu_h (-); #mpermH = #res.
        Defaults to ones.

    mpermV : array_like, optional
        Vertical magnetic permeabilities mu_v (-); #mpermV = #res.
        Defaults to ones.

    xdirect : bool, optional
        If True and source and receiver are in the same layer, the direct field
        is calculated analytically in the frequency domain, if False it is
        calculated in the wavenumber domain.
        Defaults to True.

    ht : {'fht', 'qwe'}, optional
        Flag to choose either the *Fast Hankel Transform* (FHT) or the
        *Quadrature-With-Extrapolation* (QWE) for the Hankel transform.
        Defaults to 'fht'.

    htarg : str or filter from empymod.filters or array_like, optional
        Depends on the value for `ht`:
            - If `ht` = 'fht': array containing:
              [filter, pts_per_dec]:

                - filter: string of filter name in `empymod.filters` or
                          the filter method itself.
                          (default: `empymod.filters.key_401_2009()`)
                - pts_per_dec: points per decade (only relevant if spline=True)
                               If none, standard lagged convolution is used.
                                (default: None)

            - If `ht` = 'qwe': array containing:
              [rtol, atol, nquad, maxint, pts_per_dec]:

                - rtol: relative tolerance (default: 1e-12)
                - atol: absolute tolerance (default: 1e-30)
                - nquad: order of Gaussian quadrature (default: 51)
                - maxint: maximum number of partial integral intervals
                  (default: 40)
                - pts_per_dec: points per decade (only relevant if
                  opt='spline') (default: 80)

              All are optional, you only have to maintain the order. To only
              change `nquad` to 11 and use the defaults otherwise, you can
              provide htarg=['', '', 11].

    opt : {None, 'parallel', 'spline'}, optional
        Optimization flag. Defaults to None:
            - None: Normal case, no parallelization nor interpolation is used.
            - If 'parallel', the package `numexpr` is used to evaluate the most
              expensive statements. Always check if it actually improves
              performance for a specific problem. It can speed up the
              calculation for big arrays, but will most likely be slower for
              small arrays. It will use all available cores for these specific
              statements, which all contain `Gamma` in one way or another,
              which has dimensions (#frequencies, #offsets, #layers, #lambdas),
              therefore can grow pretty big.
            - If 'spline', the *lagged convolution* or *splined* variant of the
              FHT or the *splined* version of the QWE are used. Use with
              caution and check with the non-spline version for a specific
              problem. (Can be faster, slower, or plainly wrong, as it uses
              interpolation.) If spline is set it will make use of the
              parameter pts_per_dec that can be defined in htarg. If
              pts_per_dec is not set for FHT, then the *lagged* version is
              used, else the *splined*.

        The option 'parallel' only affects speed and memory usage, whereas
        'spline' also affects precision!  Please read the note in the *README*
        documentation for more information.

    loop : {None, 'freq', 'off'}, optional
        Define if to calculate everything vectorized or if to loop over
        frequencies ('freq') or over offsets ('off'), default is None. It
        always loops over frequencies if ``ht = 'qwe'`` or if ``opt =
        'spline'``. Calculating everything vectorized is fast for few offsets
        OR for few frequencies. However, if you calculate many frequencies for
        many offsets, it might be faster to loop over frequencies. Only
        comparing the different versions will yield the answer for your
        specific problem at hand!

    verb : {0, 1, 2}, optional
        Level of verbosity, defaults to 1:
            - 0: Print nothing.
            - 1: Print warnings.
            - 2: Print warnings and information.

    Returns
    -------
    fEM : ndarray
        Frequency-domain electromagnetic field:
            - If rec is electric, returns E [V/m].
            - If rec is magnetic, returns B [T] (not H [A/m]!).

        However, the source is normalised. So for instance in the electric case
        the source strength is 1 A and its length is 1 m. So the electric field
        could also be written as [V/(A.m2)].

    Examples
    --------
    >>> import numpy as np
    >>> from empymod import frequency
    >>> src = [0, 0, 100]
    >>> rec = [np.arange(1,11)*500, np.zeros(10), 200]
    >>> depth = [0, 300, 1000, 1050]
    >>> res = [1e20, .3, 1, 50, 1]
    >>> EMfield = frequency(src, rec, depth, res, freq=1, verb=0)
    >>> print(EMfield)
    [  1.68809346e-10 -3.08303130e-10j  -8.77189179e-12 -3.76920235e-11j
      -3.46654704e-12 -4.87133683e-12j  -3.60159726e-13 -1.12434417e-12j
       1.87807271e-13 -6.21669759e-13j   1.97200208e-13 -4.38210489e-13j
       1.44134842e-13 -3.17505260e-13j   9.92770406e-14 -2.33950871e-13j
       6.75287598e-14 -1.74922886e-13j   4.62724887e-14 -1.32266600e-13j]
    """

    # === 1.  LET'S START ============
    if verb > 0:
        t0 = utils.printstartfinish(verb)

    # === 2.  CHECK INPUT ============
    inpdata = utils.fem_input(src, rec, depth, res, freq, ab, aniso, epermH,
                              epermV, mpermH, mpermV, xdirect, ht, htarg, opt,
                              loop, verb)

    # === 3. EM-FIELD CALCULATION ============
    fEM = fem(*inpdata)

    # If calc. is for only one frequency or one offset, return simple 1D array
    fEM = np.squeeze(fEM)

    # === 4.  FINISHED ============
    if verb > 0:
        utils.printstartfinish(verb, t0)

    return fEM


def time(src, rec, depth, res, time, ab=11, signal=0, aniso=None, epermH=None,
         epermV=None, mpermH=None, mpermV=None, xdirect=True, ht='fht',
         htarg=None, ft='sin', ftarg=None, opt=None, loop='off', verb=1):
    """Return the electromagnetic time-domain field.

    Calculate the electromagnetic field for various times and offsets (one
    src-rec-configuration).

    This routine runs internally the `frequency`-routine, and transforms it
    to the time domain with a Fourier transform. Most parameters are therefore
    described in the modelling-routine `frequency`: *src*, *rec*, *depth*,
    *res*, *freq*, *ab*, *aniso*, *epermH*, *epermV*, *mpermH*, *mpermV*,
    *xdirect*, *ht*, *htarg*, *opt*, *loop*, and *verb*.

    Here are only the time-domain specific input parameters described: *time*,
    *signal*, *ft*, and *ftarg*.

    Parameters
    ----------
    time : array_like
        Times t (s)

    signal : {0, 1, -1}, optional
        Source signal, default is '0':
            - -1 : Switch-off response
            - 0 : Impulse response (0 or anything that is not +/- 1)
            - +1 : Switch-on response

    ft : {'sin', 'cos', 'qwe', 'fftlog'}, optional
        Flag to choose either the Sine- or Cosine-Filter, the
        Quadrature-With-Extrapolation (QWE), or FFTLog for the Fourier
        transform.  Defaults to 'sin'.

    ftarg : str or filter from empymod.filters or array_like, optional
        Depends on the value for `ft`:
            - If `ft` = 'sin' or 'cos': array containing:
              [filter, pts_per_dec]:

                - filter: string of filter name in `empymod.filters` or
                          the filter method itself.
                          (Default: `empymod.filters.key_201_CosSin_2012()`)
                - pts_per_dec: points per decade.  If none, standard lagged
                               convolution is used. (Default: None)

            - If `ft` = 'qwe': array containing:
              [rtol, atol, nquad, maxint, pts_per_dec]:

                - rtol: relative tolerance (default: 1e-8)
                - atol: absolute tolerance (default: 1e-20)
                - nquad: order of Gaussian quadrature (default: 21)
                - maxint: maximum number of partial integral intervals
                  (default: 200)
                - pts_per_dec: points per decade (only relevant if spline=True)
                  (default: 20)

              All are optional, you only have to maintain the order. To only
              change `nquad` to 11 and use the defaults otherwise, you can
              provide ftarg=['', '', 11].

            - If `ft` = 'fftlog': array containing: [pts_per_dec, add_dec, q]:

                - pts_per_dec: sampels per decade (default: 10)
                - add_dec: additional decades [left, right] (default: [-2, 1])
                - q: exponent of power law bias (default: 0); -1 <= q <= 1

              All are optional, you only have to maintain the order. To only
              change `add_dec` to [-1, 1] and use the defaults otherwise, you
              can provide ftarg=['', [-1, 1]].

    Returns
    -------
    tEM : ndarray
        Time-domain electromagnetic field:
            - If rec is electric, returns E [V/m].
            - If rec is magnetic, returns B [T] (not H [A/m]!).

        In the case of the impulse response, the unit is further divided by
        seconds [1/s].

        However, the source is normalised. So for instance in the electric case
        the source strength is 1 A and its length is 1 m. So the electric field
        could also be written as [V/(A.m2)].

    Examples
    --------
    >>> import numpy as np
    >>> from empymod import time
    >>> src = [0, 0, 100]
    >>> rec = [3000, 0, 200]
    >>> t = np.logspace(-1,1,21)
    >>> depth = [0, 300, 1000, 1050]
    >>> res = [1e20, .3, 1, 50, 1]
    >>> EMfield = time(src, rec, depth, res, t, verb=0)
    >>> print(EMfield)
    [  4.19709394e-12   3.79932391e-12   3.36173519e-12   2.95469327e-12
       2.64314626e-12   2.49550877e-12   2.56455245e-12   2.84250641e-12
       3.22151484e-12   3.51422486e-12   3.54902799e-12   3.26936090e-12
       2.74826378e-12   2.12471621e-12   1.52728547e-12   1.03310668e-12
       6.65679090e-13   4.13283197e-13   2.49707666e-13   1.48016806e-13
       8.66001149e-14]
    """

    # === 1.  LET'S START ============
    if verb > 0:
        t0 = utils.printstartfinish(verb)

    # === 2.  CHECK INPUT ============
    inpdata = utils.tem_input(src, rec, depth, res, time, ab, signal, aniso,
                              epermH, epermV, mpermH, mpermV, xdirect, ht,
                              htarg, ft, ftarg, opt, loop, verb)

    # === 3. EM-FIELD CALCULATION AND t->f TRANSFORM ============
    tEM = tem(*inpdata)

    # If calc. is for only one time or one offset, return simple 1D array
    tEM = np.squeeze(tEM)

    # === 4.  FINISHED ============
    if verb > 0:
        utils.printstartfinish(verb, t0)

    return tEM


def gpr(src, rec, depth, res, fc=250, ab=11, gain=None, aniso=None,
        epermH=None, epermV=None, mpermH=None, mpermV=None, xdirect=True,
        ht='fht', htarg=None, opt=None, loop='off', verb=1):
    """Return the Ground-Penetrating Radar signal.

    THIS FUNCTION IS IN DEVELOPMENT, USE WITH CAUTION.

    Or in other words it is merely an example how one could calculate the
    GPR-response.  However, the currently included *FHT* and *QWE* struggle for
    these high frequencies, and another Hankel transform has to be included to
    make GPR work properly (e.g. `scipy.integrate.quad`).

    - `QWE` is slow, but does a pretty good job except for very short offsets:
      only direct wave for offset < 0.1 m, triangle-like noise at later times.
    - `FHT` is fast. Airwave, direct wave and first reflection are well
      visible, but afterwards it is very noisy.

    A lot is still hard-coded in this routine, for instance the frequency-range
    used to calculate the response.

    For input parameters see `frequency`, except for:

    Parameters
    ----------
    fc : float
        Centre frequency of GPR-signal (MHz). Sensible values are between
        10 MHz and 3000 MHz.

    gain : float
        Power of gain function. If None, no gain is applied.

    Returns
    -------
    t : array
        Times (s)
    gprEM : ndarray
        GPR response

    """
    print('* WARNING :: GPR FUNCTION IS IN DEVELOPMENT, USE WITH CAUTION')

    # === 1.  LET'S START ============
    if verb > 0:
        t0 = utils.printstartfinish(verb)

    # === 2.  CHECK INPUT ============

    # Frequency range from centre frequency
    fc *= 10**6
    freq = np.linspace(1, 2048, 2048)*10**6

    # Check the normal frequency-calculation stuff
    fdata = utils.fem_input(src, rec, depth, res, freq, ab, aniso, epermH,
                            epermV, mpermH, mpermV, xdirect, ht, htarg, opt,
                            loop, verb)

    # === 3. GPR CALCULATION ============

    # 1. Get fem responses
    fEM = fem(*fdata)

    # 2. Multiply with ricker wavelet
    cfc = -(np.r_[0, freq[:-1]]/fc)**2
    fwave = cfc*np.exp(cfc)
    fEM *= fwave[:, None]

    # 3. Carry out FFT
    tempEM = fEM[::-1, :].conj()
    tempEM = np.r_[np.zeros((1, tempEM.shape[1])), tempEM]
    dtmpEM = np.r_[tempEM, fEM[1:, :]]
    shftEM = np.fft.fftshift(dtmpEM, 0)
    ifftEM = np.fft.ifft(shftEM, axis=0).real
    nfreq = 2*freq.size
    dfreq = freq[1]-freq[0]
    gprEM = nfreq*np.fft.fftshift(ifftEM*dfreq, 0)
    dt = 1/(nfreq*dfreq)

    # 4. Apply gain
    t = np.linspace(-nfreq/2, nfreq/2-1, nfreq)*dt
    if gain:
        gprEM *= (1 + np.abs((t*10**9)**gain))[:, None]

    # === 4.  FINISHED ============
    if verb > 0:
        utils.printstartfinish(verb, t0)

    return t[2048:], gprEM[2048:, :].real


def wavenumber(src, rec, depth, res, freq, wavenumber, ab=11, aniso=None,
               epermH=None, epermV=None, mpermH=None, mpermV=None,
               xdirect=True, verb=1):
    """Return the electromagnetic wavenumber-domain field.

    THIS FUNCTION IS IN DEVELOPMENT, USE WITH CAUTION.

    Or rather, it is for development purposes, to easily get the wavenumber
    result with the required input checks.

    For input parameters see `frequency`, except for:

    Parameters
    ----------
    wavenumber : array
        Wavenumbers lambda (1/m)

    Returns
    -------
    PJ0, PJ1, PJ0b : array
        Wavenumber domain EM responses.
        - PJ0 is angle independent, PJ1 and PJ0b depend on the angle.
        - PJ0 and PJ0b are J_0 functions, PJ1 is a J_1 function.

    """
    print('* WARNING :: WAVENUMBER FUNCTION IS IN DEVELOPMENT, USE WITH ' +
          'CAUTION')

    # === 1.  LET'S START ============
    if verb > 0:
        t0 = utils.printstartfinish(verb)

    # === 2.  CHECK INPUT ============
    # Check data with dummies for:
    # => ht, htarg, opt, loop
    inpdata = utils.fem_input(src, rec, depth, res, freq, ab, aniso, epermH,
                              epermV, mpermH, mpermV, xdirect, ht='fht',
                              htarg=None, opt=None, loop=None, verb=verb)

    ab, _, _, _, angle, zsrc, zrec = inpdata[:7]
    lsrc, lrec, depth, _, etaH, etaV, zetaH, zetaV = inpdata[7:15]
    xdirect, _, _, _, _, _, msrc, mrec, _, _ = inpdata[15:]

    # === 3. EM-FIELD CALCULATION ============
    PJ0, PJ1, PJ0b = kernel.wavenumber(zsrc, zrec, lsrc, lrec, depth, etaH,
                                       etaV, zetaH, zetaV,
                                       np.atleast_2d(wavenumber), ab, xdirect,
                                       msrc, mrec, False)

    PJ0 = np.squeeze(PJ0)
    PJ1 = np.squeeze(PJ1*wavenumber)
    PJ0b = np.squeeze(PJ0b)

    # === 4.  FINISHED ============
    if verb > 0:
        utils.printstartfinish(verb, t0)

    return PJ0, PJ1, PJ0b


# Core modelling routines

def fem(ab, xco, yco, off, angle, zsrc, zrec, lsrc, lrec, depth, freq, etaH,
        etaV, zetaH, zetaV, xdirect, isfullspace, ht, htarg, use_spline,
        use_ne_eval, msrc, mrec, loop_freq, loop_off):
    """Return the electromagnetic frequency-domain response.

    This function is called from one of the above modelling routines. No
    input-check is carried out here. See the main description of :mod:`model`
    for information regarding input and output parameters.

    This function can be directly used if you are sure the provided input is in
    the correct format. This is useful for inversion routines and similar, as
    it can speed-up the calculation by omitting input-checks.

    """
    # Preallocate array
    fEM = np.zeros((freq.size, off.size), dtype=complex)

    # If <ab> = 36 (or 63), fEM-field is zero
    if ab in [36, ]:
        return fEM

    # Get full-space-solution if model is a full-space or
    # if src and rec are in the same layer and xdirect=True.
    if isfullspace or (lsrc == lrec and xdirect):
        fEM += kernel.fullspace(xco, yco, off, zsrc, zrec, etaH[:, lrec],
                                etaV[:, lrec], zetaH[:, lrec], zetaV[:, lrec],
                                ab, msrc, mrec)

    # If not full-space calculate fEM-field
    if not isfullspace:
        calc = getattr(transform, ht)
        if loop_freq:

            for i in range(freq.size):
                fEM[None, i, :] += calc(zsrc, zrec, lsrc, lrec, off, angle,
                                        depth, ab, etaH[None, i, :],
                                        etaV[None, i, :], zetaH[None, i, :],
                                        zetaV[None, i, :], xdirect, htarg,
                                        use_spline, use_ne_eval, msrc, mrec)
        elif loop_off:
            for i in range(off.size):
                fEM[:, None, i] += calc(zsrc, zrec, lsrc, lrec, off[None, i],
                                        angle[None, i], depth, ab, etaH, etaV,
                                        zetaH, zetaV, xdirect, htarg,
                                        use_spline, use_ne_eval, msrc, mrec)
        else:
            fEM += calc(zsrc, zrec, lsrc, lrec, off, angle, depth, ab, etaH,
                        etaV, zetaH, zetaV, xdirect, htarg, use_spline,
                        use_ne_eval, msrc, mrec)

    return fEM


def tem(ab, xco, yco, off, angle, zsrc, zrec, lsrc, lrec, depth, freq, etaH,
        etaV, zetaH, zetaV, xdirect, isfullspace, ht, htarg, time, signal, ft,
        ftarg, use_spline, use_ne_eval, msrc, mrec, loop_freq, loop_off):
    """Return the electromagnetic time-domain response.

    This function is called from one of the above modelling routines. No
    input-check is carried out here. See the main description of :mod:`model`
    for information regarding input and output parameters.

    This function can be directly used if you are sure the provided input is in
    the correct format. This is useful for inversion routines and similar, as
    it can speed-up the calculation by omitting input-checks.

    """
    # 1. Get fem responses
    fEM = fem(ab, xco, yco, off, angle, zsrc, zrec, lsrc, lrec, depth, freq,
              etaH, etaV, zetaH, zetaV, xdirect, isfullspace, ht, htarg,
              use_spline, use_ne_eval, msrc, mrec, loop_freq, loop_off)

    # 2. Scale frequencies if switch-on/off response
    # Step function for causal times is like a unit fct, therefore an impulse
    # in frequency domain
    if signal in [-1, 1]:
        fEM *= signal/(2j*np.pi*freq[:, None])

    # 3. f->t transform
    tEM = np.zeros((time.size, off.size))
    for i in range(off.size):
        tEM[:, i] += getattr(transform, ft)(fEM[:, i], time, freq, ftarg)

    return tEM*2/np.pi  # Scaling from Fourier transform
