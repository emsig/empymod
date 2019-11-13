r"""

:mod:`model` -- Model EM-responses
==================================

EM-modelling routines. The implemented routines might not be the fastest
solution to your specific problem. Use these routines as template to create
your own, problem-specific modelling routine!

Principal routines:
    - ``bipole``
    - ``dipole``
    - ``loop``

The main routine is ``bipole``, which can model bipole source(s) and bipole
receiver(s) of arbitrary direction, for electric or magnetic sources and
receivers, both in frequency and in time. A subset of ``bipole`` is ``dipole``,
which models infinitesimal small dipoles along the principal axes x, y, and z.
The third routine, ``loop``, can be used if the source or the receivers are
loops instead of dipoles.

Further routines are:

    - ``analytical``: Calculate analytical fullspace and halfspace solutions.
    - ``dipole_k``:   Calculate the electromagnetic wavenumber-domain solution.
    - ``gpr``:        Calculate the Ground-Penetrating Radar (GPR) response.

The ``dipole_k`` routine can be used if you are interested in the
wavenumber-domain result, without Hankel nor Fourier transform. It calls
straight the ``kernel``. The ``gpr``-routine convolves the frequency-domain
result with a wavelet, and applies a gain to the time-domain result. This
function is still experimental.

The modelling routines make use of the following two core routines:
    - ``fem``: Calculate wavenumber-domain electromagnetic field and carry out
               the Hankel transform to the frequency domain.
    - ``tem``: Carry out the Fourier transform to time domain after ``fem``.

"""
# Copyright 2016-2019 The empymod Developers.
#
# This file is part of empymod.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License.  You may obtain a copy
# of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  See the
# License for the specific language governing permissions and limitations under
# the License.


import warnings
import numpy as np

from empymod import kernel, transform
from empymod.utils import (
        check_time, check_time_only, check_model, check_frequency,
        check_hankel, check_opt, check_dipole, check_bipole, check_ab,
        check_solution, get_abs, get_geo_fact, get_azm_dip, get_off_ang,
        get_layer_nr, printstartfinish, conv_warning, spline_backwards_hankel,
        EMArray)

__all__ = ['bipole', 'dipole', 'loop', 'analytical', 'gpr', 'dipole_k', 'fem',
           'tem', 'wavenumber']


def bipole(src, rec, depth, res, freqtime, signal=None, aniso=None,
           epermH=None, epermV=None, mpermH=None, mpermV=None, msrc=False,
           srcpts=1, mrec=False, recpts=1, strength=0, xdirect=False,
           ht='fht', htarg=None, ft='sin', ftarg=None, opt=None, loop=None,
           verb=2):
    r"""Return EM fields due to arbitrary rotated, finite length EM dipoles.

    Calculate the electromagnetic frequency- or time-domain field due to
    arbitrary rotated, finite electric or magnetic bipole sources, measured by
    arbitrary rotated, finite electric or magnetic bipole receivers. By
    default, the electromagnetic response is normalized to source and receiver
    of 1 m length, and source strength of 1 A.


    See Also
    --------
    dipole : EM fields due to infinitesimal small EM dipoles.
    loop : EM fields due to a magnetic source loop.


    Parameters
    ----------
    src, rec : list of floats or arrays
        Source and receiver coordinates (m):
            - [x0, x1, y0, y1, z0, z1] (bipole of finite length)
            - [x, y, z, azimuth, dip]  (dipole, infinitesimal small)

        Dimensions:
            - The coordinates x, y, and z (dipole) or x0, x1, y0, y1, z0, and
              z1 (bipole) can be single values or arrays.
            - The variables x and y (dipole) or x0, x1, y0, and y1 (bipole)
              must have the same dimensions.
            - The variable z (dipole) or z0 and z1 (bipole) must either be
              single values or having the same dimension as the other
              coordinates.
            - The variables azimuth and dip must be single values. If they have
              different angles, you have to use the bipole-method (with
              srcpts/recpts = 1, so it is calculated as dipoles).

        Angles (coordinate system is left-handed, positive z down
        (East-North-Depth):

            - azimuth (°): horizontal deviation from x-axis, anti-clockwise.
            - dip (°): vertical deviation from xy-plane downwards.

        Sources or receivers placed on a layer interface are considered in the
        upper layer.

    depth : list
        Absolute layer interfaces z (m); #depth = #res - 1
        (excluding +/- infinity).

    res : array_like
        Horizontal resistivities rho_h (Ohm.m); #res = #depth + 1.

        Alternatively, res can be a dictionary. See the main manual of empymod
        too see how to exploit this hook to re-calculate etaH, etaV, zetaH, and
        zetaV, which can be used to, for instance, use the Cole-Cole model for
        IP.

    freqtime : array_like
        Frequencies f (Hz) if ``signal`` == None, else times t (s); (f, t > 0).

    signal : {None, 0, 1, -1}, optional
        Source signal, default is None:
            - None: Frequency-domain response
            - -1 : Switch-off time-domain response
            - 0 : Impulse time-domain response
            - +1 : Switch-on time-domain response

    aniso : array_like, optional
        Anisotropies lambda = sqrt(rho_v/rho_h) (-); #aniso = #res.
        Defaults to ones.

    epermH, epermV : array_like, optional
        Relative horizontal/vertical electric permittivities
        epsilon_h/epsilon_v (-);
        #epermH = #epermV = #res. Default is ones.

    mpermH, mpermV : array_like, optional
        Relative horizontal/vertical magnetic permeabilities mu_h/mu_v (-);
        #mpermH = #mpermV = #res. Default is ones.

    msrc, mrec : boolean, optional
        If True, source/receiver (msrc/mrec) is magnetic, else electric.
        Default is False.

    srcpts, recpts : int, optional
        Number of integration points for bipole source/receiver, default is 1:
            - srcpts/recpts < 3  : bipole, but calculated as dipole at centre
            - srcpts/recpts >= 3 : bipole

    strength : float, optional
        Source strength (A):
          - If 0, output is normalized to source and receiver of 1 m length,
            and source strength of 1 A.
          - If != 0, output is returned for given source and receiver length,
            and source strength.

        Default is 0.

    xdirect : bool or None, optional
        Direct field calculation (only if src and rec are in the same layer):
          - If True, direct field is calculated analytically in the frequency
            domain.
          - If False, direct field is calculated in the wavenumber domain.
          - If None, direct field is excluded from the calculation, and only
            reflected fields are returned (secondary field).

        Defaults to False.

    ht : {'fht', 'qwe', 'quad'}, optional
        Flag to choose either the *Digital Linear Filter* method (FHT, *Fast
        Hankel Transform*), the *Quadrature-With-Extrapolation* (QWE), or a
        simple *Quadrature* (QUAD) for the Hankel transform.  Defaults to
        'fht'.

    htarg : dict or list, optional
        Depends on the value for ``ht``:
            - If ``ht`` = 'fht': [fhtfilt, pts_per_dec]:

                - fhtfilt: string of filter name in ``empymod.filters`` or
                           the filter method itself.
                           (default: ``empymod.filters.key_201_2009()``)
                - pts_per_dec: points per decade; (default: 0)
                    - If 0: Standard DLF.
                    - If < 0: Lagged Convolution DLF.
                    - If > 0: Splined DLF

            - If ``ht`` = 'qwe': [rtol, atol, nquad, maxint, pts_per_dec,
                                diff_quad, a, b, limit]:

                - rtol: relative tolerance (default: 1e-12)
                - atol: absolute tolerance (default: 1e-30)
                - nquad: order of Gaussian quadrature (default: 51)
                - maxint: maximum number of partial integral intervals
                          (default: 40)
                - pts_per_dec: points per decade; (default: 0)
                    - If 0, no interpolation is used.
                    - If > 0, interpolation is used.

                - diff_quad: criteria when to swap to QUAD (only relevant if
                  opt='spline') (default: 100)
                - a: lower limit for QUAD (default: first interval from QWE)
                - b: upper limit for QUAD (default: last interval from QWE)
                - limit: limit for quad (default: maxint)

            - If ``ht`` = 'quad': [atol, rtol, limit, lmin, lmax, pts_per_dec]:

                - rtol: relative tolerance (default: 1e-12)
                - atol: absolute tolerance (default: 1e-20)
                - limit: An upper bound on the number of subintervals used in
                  the adaptive algorithm (default: 500)
                - lmin: Minimum wavenumber (default 1e-6)
                - lmax: Maximum wavenumber (default 0.1)
                - pts_per_dec: points per decade (default: 40)

        The values can be provided as dict with the keywords, or as list.
        However, if provided as list, you have to follow the order given above.
        A few examples, assuming ``ht`` = ``qwe``:

            - Only changing rtol:
                {'rtol': 1e-4} or [1e-4] or 1e-4
            - Changing rtol and nquad:
                {'rtol': 1e-4, 'nquad': 101} or [1e-4, '', 101]
            - Only changing diff_quad:
                {'diffquad': 10} or ['', '', '', '', '', 10]

    ft : {'sin', 'cos', 'qwe', 'fftlog', 'fft'}, optional
        Only used if ``signal`` != None. Flag to choose either the Digital
        Linear Filter method (Sine- or Cosine-Filter), the
        Quadrature-With-Extrapolation (QWE), the FFTLog, or the FFT for the
        Fourier transform.  Defaults to 'sin'.

    ftarg : dict or list, optional
        Only used if ``signal`` !=None. Depends on the value for ``ft``:
            - If ``ft`` = 'sin' or 'cos': [fftfilt, pts_per_dec]:

                - fftfilt: string of filter name in ``empymod.filters`` or
                           the filter method itself.
                           (Default: ``empymod.filters.key_201_CosSin_2012()``)
                - pts_per_dec: points per decade; (default: -1)
                    - If 0: Standard DLF.
                    - If < 0: Lagged Convolution DLF.
                    - If > 0: Splined DLF


            - If ``ft`` = 'qwe': [rtol, atol, nquad, maxint, pts_per_dec]:

                - rtol: relative tolerance (default: 1e-8)
                - atol: absolute tolerance (default: 1e-20)
                - nquad: order of Gaussian quadrature (default: 21)
                - maxint: maximum number of partial integral intervals
                          (default: 200)
                - pts_per_dec: points per decade (default: 20)
                - diff_quad: criteria when to swap to QUAD (default: 100)
                - a: lower limit for QUAD (default: first interval from QWE)
                - b: upper limit for QUAD (default: last interval from QWE)
                - limit: limit for quad (default: maxint)

            - If ``ft`` = 'fftlog': [pts_per_dec, add_dec, q]:

                - pts_per_dec: sampels per decade (default: 10)
                - add_dec: additional decades [left, right] (default: [-2, 1])
                - q: exponent of power law bias (default: 0); -1 <= q <= 1

            - If ``ft`` = 'fft': [dfreq, nfreq, ntot]:

                - dfreq: Linear step-size of frequencies (default: 0.002)
                - nfreq: Number of frequencies (default: 2048)
                - ntot:  Total number for FFT; difference between nfreq and
                         ntot is padded with zeroes. This number is ideally a
                         power of 2, e.g. 2048 or 4096 (default: nfreq).
                - pts_per_dec : points per decade (default: None)

                Padding can sometimes improve the result, not always. The
                default samples from 0.002 Hz - 4.096 Hz. If pts_per_dec is set
                to an integer, calculated frequencies are logarithmically
                spaced with the given number per decade, and then interpolated
                to yield the required frequencies for the FFT.

        The values can be provided as dict with the keywords, or as list.
        However, if provided as list, you have to follow the order given above.
        See ``htarg`` for a few examples.

    opt : {None, 'parallel'}, optional
        Optimization flag. Defaults to None:
            - None: Normal case, no parallelization nor interpolation is used.
            - If 'parallel', the package ``numexpr`` is used to evaluate the
              most expensive statements. Always check if it actually improves
              performance for a specific problem. It can speed up the
              calculation for big arrays, but will most likely be slower for
              small arrays. It will use all available cores for these specific
              statements, which all contain ``Gamma`` in one way or another,
              which has dimensions (#frequencies, #offsets, #layers, #lambdas),
              therefore can grow pretty big. The module ``numexpr`` uses by
              default all available cores up to a maximum of 8. You can change
              this behaviour to your desired number of threads ``nthreads``
              with ``numexpr.set_num_threads(nthreads)``.
            - The value 'spline' is deprecated and will be removed. See
              ``htarg`` instead for the interpolated versions.

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

    verb : {0, 1, 2, 3, 4}, optional
        Level of verbosity, default is 2:
            - 0: Print nothing.
            - 1: Print warnings.
            - 2: Print additional runtime and kernel calls
            - 3: Print additional start/stop, condensed parameter information.
            - 4: Print additional full parameter information


    Returns
    -------
    EM : EMAarray, (nfreqtime, nrec, nsrc)
        Frequency- or time-domain EM field (depending on ``signal``):
            - If rec is electric, returns E [V/m].
            - If rec is magnetic, returns H [A/m].

        EMArray is a subclassed ndarray with ``.pha`` and ``.amp`` attributes
        (only relevant for frequency-domain data).

        The shape of EM is (nfreqtime, nrec, nsrc). However, single dimensions
        are removed.


    Examples
    --------
    >>> import numpy as np
    >>> from empymod import bipole
    >>> # x-directed bipole source: x0, x1, y0, y1, z0, z1
    >>> src = [-50, 50, 0, 0, 100, 100]
    >>> # x-directed dipole receiver-array: x, y, z, azimuth, dip
    >>> rec = [np.arange(1, 11)*500, np.zeros(10), 200, 0, 0]
    >>> # layer boundaries
    >>> depth = [0, 300, 1000, 1050]
    >>> # layer resistivities
    >>> res = [1e20, .3, 1, 50, 1]
    >>> # Frequency
    >>> freq = 1
    >>> # Calculate electric field due to an electric source at 1 Hz.
    >>> # [msrc = mrec = False (default)]
    >>> EMfield = bipole(src, rec, depth, res, freq, verb=4)
    :: empymod START  ::
    ~
       depth       [m] :  0 300 1000 1050
       res     [Ohm.m] :  1E+20 0.3 1 50 1
       aniso       [-] :  1 1 1 1 1
       epermH      [-] :  1 1 1 1 1
       epermV      [-] :  1 1 1 1 1
       mpermH      [-] :  1 1 1 1 1
       mpermV      [-] :  1 1 1 1 1
       frequency  [Hz] :  1
       Hankel          :  DLF (Fast Hankel Transform)
         > Filter      :  Key 201 (2009)
         > DLF type    :  Standard
       Kernel Opt.     :  None
       Loop over       :  None (all vectorized)
       Source(s)       :  1 bipole(s)
         > intpts      :  1 (as dipole)
         > length  [m] :  100
         > x_c     [m] :  0
         > y_c     [m] :  0
         > z_c     [m] :  100
         > azimuth [°] :  0
         > dip     [°] :  0
       Receiver(s)     :  10 dipole(s)
         > x       [m] :  500 - 5000 : 10  [min-max; #]
                       :  500 1000 1500 2000 2500 3000 3500 4000 4500 5000
         > y       [m] :  0 - 0 : 10  [min-max; #]
                       :  0 0 0 0 0 0 0 0 0 0
         > z       [m] :  200
         > azimuth [°] :  0
         > dip     [°] :  0
       Required ab's   :  11
    ~
    :: empymod END; runtime = 0:00:00.005536 :: 1 kernel call(s)
    ~
    >>> print(EMfield)
    [  1.68809346e-10 -3.08303130e-10j  -8.77189179e-12 -3.76920235e-11j
      -3.46654704e-12 -4.87133683e-12j  -3.60159726e-13 -1.12434417e-12j
       1.87807271e-13 -6.21669759e-13j   1.97200208e-13 -4.38210489e-13j
       1.44134842e-13 -3.17505260e-13j   9.92770406e-14 -2.33950871e-13j
       6.75287598e-14 -1.74922886e-13j   4.62724887e-14 -1.32266600e-13j]

    """

    # === 1.  LET'S START ============
    t0 = printstartfinish(verb)

    # === 2.  CHECK INPUT ============

    # Backwards compatibility
    htarg, opt = spline_backwards_hankel(ht, htarg, opt)

    # Check times and Fourier Transform arguments and get required frequencies
    if signal is None:
        freq = freqtime
    else:
        time, freq, ft, ftarg = check_time(freqtime, signal, ft, ftarg, verb)

    # Check layer parameters
    model = check_model(depth, res, aniso, epermH, epermV, mpermH, mpermV,
                        xdirect, verb)
    depth, res, aniso, epermH, epermV, mpermH, mpermV, isfullspace = model

    # Check frequency => get etaH, etaV, zetaH, and zetaV
    frequency = check_frequency(freq, res, aniso, epermH, epermV, mpermH,
                                mpermV, verb)
    freq, etaH, etaV, zetaH, zetaV = frequency

    # Update etaH/etaV and zetaH/zetaV according to user-provided model
    if isinstance(res, dict) and 'func_eta' in res:
        etaH, etaV = res['func_eta'](res, locals())
    if isinstance(res, dict) and 'func_zeta' in res:
        zetaH, zetaV = res['func_zeta'](res, locals())

    # Check Hankel transform parameters
    ht, htarg = check_hankel(ht, htarg, verb)

    # Check optimization
    use_ne_eval, loop_freq, loop_off = check_opt(opt, loop, ht, htarg, verb)

    # Check src and rec, get flags if dipole or not
    # nsrcz/nrecz are number of unique src/rec-pole depths
    src, nsrc, nsrcz, srcdipole = check_bipole(src, 'src')
    rec, nrec, nrecz, recdipole = check_bipole(rec, 'rec')

    # === 3. EM-FIELD CALCULATION ============

    # Pre-allocate output EM array
    EM = np.zeros((freq.size, nrec*nsrc), dtype=etaH.dtype)

    # Initialize kernel count, conv (only for QWE)
    # (how many times the wavenumber-domain kernel was calld)
    kcount = 0
    conv = True

    # Define some indeces
    isrc = int(nsrc/nsrcz)  # this is either 1 or nsrc
    irec = int(nrec/nrecz)  # this is either 1 or nrec
    isrz = int(isrc*irec)   # this is either 1, nsrc, nrec, or nsrc*nrec

    # The kernel handles only 1 ab with one srcz-recz combination at once.
    # Hence we have to loop over every different depth of src or rec, and
    # over all required ab's.
    for isz in range(nsrcz):  # Loop over source depths

        # Get this source
        srcazmdip = get_azm_dip(src, isz, nsrcz, srcpts, srcdipole, strength,
                                'src', verb)
        tsrc, srcazm, srcdip, srcg_w, srcpts, src_w = srcazmdip

        for irz in range(nrecz):  # Loop over receiver depths

            # Get this receiver
            recazmdip = get_azm_dip(rec, irz, nrecz, recpts, recdipole,
                                    strength, 'rec', verb)
            trec, recazm, recdip, recg_w, recpts, rec_w = recazmdip

            # Get required ab's
            ab_calc = get_abs(msrc, mrec, srcazm, srcdip, recazm, recdip, verb)

            # Pre-allocate temporary source-EM array for integration loop
            sEM = np.zeros((freq.size, isrz), dtype=etaH.dtype)

            for isg in range(srcpts):  # Loop over src integration points

                # This integration source
                tisrc = [tsrc[0][isg::srcpts], tsrc[1][isg::srcpts],
                         tsrc[2][isg]]

                # Get layer number in which src resides
                lsrc, zsrc = get_layer_nr(tisrc, depth)

                # Pre-allocate temporary receiver EM arrays for integr. loop
                rEM = np.zeros((freq.size, isrz), dtype=etaH.dtype)

                for irg in range(recpts):  # Loop over rec integration pts
                    # Note, if source or receiver is a bipole, but horizontal
                    # (dip=0), then calculation could be sped up by not looping
                    # over the bipole elements, but calculate it all in one go.

                    # This integration receiver
                    tirec = [trec[0][irg::recpts], trec[1][irg::recpts],
                             trec[2][irg]]

                    # Get src-rec offsets and angles
                    off, angle = get_off_ang(tisrc, tirec, isrc, irec, verb)

                    # Get layer number in which rec resides
                    lrec, zrec = get_layer_nr(tirec, depth)

                    # Gather variables
                    finp = (off, angle, zsrc, zrec, lsrc, lrec, depth, freq,
                            etaH, etaV, zetaH, zetaV, xdirect, isfullspace, ht,
                            htarg, use_ne_eval, msrc, mrec, loop_freq,
                            loop_off, conv)

                    # Pre-allocate temporary EM array for ab-loop
                    abEM = np.zeros((freq.size, isrz), dtype=etaH.dtype)

                    for iab in ab_calc:  # Loop over required ab's

                        # Carry-out the frequency-domain calculation
                        out = fem(iab, *finp)

                        # Get geometrical scaling factor
                        tfact = get_geo_fact(iab, srcazm, srcdip, recazm,
                                             recdip, msrc, mrec)

                        # Add field to EM with geometrical factor
                        abEM += out[0]*np.squeeze(tfact)

                        # Update kernel count
                        kcount += out[1]

                        # Update conv (QWE convergence)
                        conv *= out[2]

                    # Add this receiver element, with weight from integration
                    rEM += abEM*recg_w[irg]

                # Add this source element, with weight from integration
                sEM += rEM*srcg_w[isg]

            # Scale signal for src-strength and src/rec-lengths
            src_rec_w = 1
            if strength > 0:
                src_rec_w *= np.repeat(src_w, irec)
                src_rec_w *= np.tile(rec_w, isrc)
            sEM *= src_rec_w

            # Add this src-rec signal
            if nrec == nrecz:
                if nsrc == nsrcz:  # Case 1: Looped over each src and each rec
                    EM[:, isz*nrec+irz:isz*nrec+irz+1] = sEM
                else:              # Case 2: Looped over each rec
                    EM[:, irz:nsrc*nrec:nrec] = sEM
            else:
                if nsrc == nsrcz:  # Case 3: Looped over each src
                    EM[:, isz*nrec:nrec*(isz+1)] = sEM
                else:              # Case 4: All in one go
                    EM = sEM

    # In case of QWE/QUAD, print Warning if not converged
    conv_warning(conv, htarg, 'Hankel', verb)

    # Do f->t transform if required
    if signal is not None:
        EM, conv = tem(EM, EM[0, :], freq, time, signal, ft, ftarg)

        # In case of QWE/QUAD, print Warning if not converged
        conv_warning(conv, ftarg, 'Fourier', verb)

    # Reshape for number of sources
    EM = np.squeeze(EM.reshape((-1, nrec, nsrc), order='F'))

    # === 4.  FINISHED ============
    printstartfinish(verb, t0, kcount)

    return EMArray(EM)


def dipole(src, rec, depth, res, freqtime, signal=None, ab=11, aniso=None,
           epermH=None, epermV=None, mpermH=None, mpermV=None, xdirect=False,
           ht='fht', htarg=None, ft='sin', ftarg=None, opt=None, loop=None,
           verb=2):
    r"""Return EM fields due to infinitesimal small EM dipoles.

    Calculate the electromagnetic frequency- or time-domain field due to
    infinitesimal small electric or magnetic dipole source(s), measured by
    infinitesimal small electric or magnetic dipole receiver(s); sources and
    receivers are directed along the principal directions x, y, or z, and all
    sources are at the same depth, as well as all receivers are at the same
    depth.

    Use the functions ``bipole`` to calculate dipoles with arbitrary angles or
    bipoles of finite length and arbitrary angle.

    The function ``dipole`` could be replaced by ``bipole`` (all there is to do
    is translate ``ab`` into ``msrc``, ``mrec``, ``azimuth``'s and ``dip``'s).
    However, ``dipole`` is kept separately to serve as an example of a simple
    modelling routine that can serve as a template.


    See Also
    --------
    bipole : EM fields due to arbitrary rotated, finite length EM dipoles.
    loop : EM fields due to a magnetic source loop.


    Parameters
    ----------
    src, rec : list of floats or arrays
        Source and receiver coordinates (m): [x, y, z].
        The x- and y-coordinates can be arrays, z is a single value.
        The x- and y-coordinates must have the same dimension.

        Sources or receivers placed on a layer interface are considered in the
        upper layer.

    depth : list
        Absolute layer interfaces z (m); #depth = #res - 1
        (excluding +/- infinity).

    res : array_like
        Horizontal resistivities rho_h (Ohm.m); #res = #depth + 1.

        Alternatively, res can be a dictionary. See the main manual of empymod
        too see how to exploit this hook to re-calculate etaH, etaV, zetaH, and
        zetaV, which can be used to, for instance, use the Cole-Cole model for
        IP.

    freqtime : array_like
        Frequencies f (Hz) if ``signal`` == None, else times t (s); (f, t > 0).

    signal : {None, 0, 1, -1}, optional
        Source signal, default is None:
            - None: Frequency-domain response
            - -1 : Switch-off time-domain response
            - 0 : Impulse time-domain response
            - +1 : Switch-on time-domain response

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

    epermH, epermV : array_like, optional
        Relative horizontal/vertical electric permittivities
        epsilon_h/epsilon_v (-);
        #epermH = #epermV = #res. Default is ones.

    mpermH, mpermV : array_like, optional
        Relative horizontal/vertical magnetic permeabilities mu_h/mu_v (-);
        #mpermH = #mpermV = #res. Default is ones.

    xdirect : bool or None, optional
        Direct field calculation (only if src and rec are in the same layer):
          - If True, direct field is calculated analytically in the frequency
            domain.
          - If False, direct field is calculated in the wavenumber domain.
          - If None, direct field is excluded from the calculation, and only
            reflected fields are returned (secondary field).

        Defaults to False.

    ht : {'fht', 'qwe', 'quad'}, optional
        Flag to choose either the *Digital Linear Filter* method (FHT, *Fast
        Hankel Transform*), the *Quadrature-With-Extrapolation* (QWE), or a
        simple *Quadrature* (QUAD) for the Hankel transform.  Defaults to
        'fht'.

    htarg : dict or list, optional
        Depends on the value for ``ht``:
            - If ``ht`` = 'fht': [fhtfilt, pts_per_dec]:

                - fhtfilt: string of filter name in ``empymod.filters`` or
                           the filter method itself.
                           (default: ``empymod.filters.key_201_2009()``)
                - pts_per_dec: points per decade; (default: 0)
                    - If 0: Standard DLF.
                    - If < 0: Lagged Convolution DLF.
                    - If > 0: Splined DLF

            - If ``ht`` = 'qwe': [rtol, atol, nquad, maxint, pts_per_dec,
                                diff_quad, a, b, limit]:

                - rtol: relative tolerance (default: 1e-12)
                - atol: absolute tolerance (default: 1e-30)
                - nquad: order of Gaussian quadrature (default: 51)
                - maxint: maximum number of partial integral intervals
                          (default: 40)
                - pts_per_dec: points per decade; (default: 0)
                    - If 0, no interpolation is used.
                    - If > 0, interpolation is used.

                - diff_quad: criteria when to swap to QUAD (only relevant if
                  opt='spline') (default: 100)
                - a: lower limit for QUAD (default: first interval from QWE)
                - b: upper limit for QUAD (default: last interval from QWE)
                - limit: limit for quad (default: maxint)

            - If ``ht`` = 'quad': [atol, rtol, limit, lmin, lmax, pts_per_dec]:

                - rtol: relative tolerance (default: 1e-12)
                - atol: absolute tolerance (default: 1e-20)
                - limit: An upper bound on the number of subintervals used in
                  the adaptive algorithm (default: 500)
                - lmin: Minimum wavenumber (default 1e-6)
                - lmax: Maximum wavenumber (default 0.1)
                - pts_per_dec: points per decade (default: 40)

        The values can be provided as dict with the keywords, or as list.
        However, if provided as list, you have to follow the order given above.
        A few examples, assuming ``ht`` = ``qwe``:

            - Only changing rtol:
                {'rtol': 1e-4} or [1e-4] or 1e-4
            - Changing rtol and nquad:
                {'rtol': 1e-4, 'nquad': 101} or [1e-4, '', 101]
            - Only changing diff_quad:
                {'diffquad': 10} or ['', '', '', '', '', 10]

    ft : {'sin', 'cos', 'qwe', 'fftlog', 'fft'}, optional
        Only used if ``signal`` != None. Flag to choose either the Digital
        Linear Filter method (Sine- or Cosine-Filter), the
        Quadrature-With-Extrapolation (QWE), the FFTLog, or the FFT for the
        Fourier transform.  Defaults to 'sin'.

    ftarg : dict or list, optional
        Only used if ``signal`` !=None. Depends on the value for ``ft``:
            - If ``ft`` = 'sin' or 'cos': [fftfilt, pts_per_dec]:

                - fftfilt: string of filter name in ``empymod.filters`` or
                           the filter method itself.
                           (Default: ``empymod.filters.key_201_CosSin_2012()``)
                - pts_per_dec: points per decade; (default: -1)
                    - If 0: Standard DLF.
                    - If < 0: Lagged Convolution DLF.
                    - If > 0: Splined DLF

            - If ``ft`` = 'qwe': [rtol, atol, nquad, maxint, pts_per_dec]:

                - rtol: relative tolerance (default: 1e-8)
                - atol: absolute tolerance (default: 1e-20)
                - nquad: order of Gaussian quadrature (default: 21)
                - maxint: maximum number of partial integral intervals
                          (default: 200)
                - pts_per_dec: points per decade (default: 20)
                - diff_quad: criteria when to swap to QUAD (default: 100)
                - a: lower limit for QUAD (default: first interval from QWE)
                - b: upper limit for QUAD (default: last interval from QWE)
                - limit: limit for quad (default: maxint)

            - If ``ft`` = 'fftlog': [pts_per_dec, add_dec, q]:

                - pts_per_dec: sampels per decade (default: 10)
                - add_dec: additional decades [left, right] (default: [-2, 1])
                - q: exponent of power law bias (default: 0); -1 <= q <= 1

            - If ``ft`` = 'fft': [dfreq, nfreq, ntot]:

                - dfreq: Linear step-size of frequencies (default: 0.002)
                - nfreq: Number of frequencies (default: 2048)
                - ntot:  Total number for FFT; difference between nfreq and
                         ntot is padded with zeroes. This number is ideally a
                         power of 2, e.g. 2048 or 4096 (default: nfreq).
                - pts_per_dec : points per decade (default: None)

                Padding can sometimes improve the result, not always. The
                default samples from 0.002 Hz - 4.096 Hz. If pts_per_dec is set
                to an integer, calculated frequencies are logarithmically
                spaced with the given number per decade, and then interpolated
                to yield the required frequencies for the FFT.

        The values can be provided as dict with the keywords, or as list.
        However, if provided as list, you have to follow the order given above.
        See ``htarg`` for a few examples.

    opt : {None, 'parallel'}, optional
        Optimization flag. Defaults to None:
            - None: Normal case, no parallelization nor interpolation is used.
            - If 'parallel', the package ``numexpr`` is used to evaluate the
              most expensive statements. Always check if it actually improves
              performance for a specific problem. It can speed up the
              calculation for big arrays, but will most likely be slower for
              small arrays. It will use all available cores for these specific
              statements, which all contain ``Gamma`` in one way or another,
              which has dimensions (#frequencies, #offsets, #layers, #lambdas),
              therefore can grow pretty big. The module ``numexpr`` uses by
              default all available cores up to a maximum of 8. You can change
              this behaviour to your desired number of threads ``nthreads``
              with ``numexpr.set_num_threads(nthreads)``.
            - The value 'spline' is deprecated and will be removed. See
              ``htarg`` instead for the interpolated versions.

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

    verb : {0, 1, 2, 3, 4}, optional
        Level of verbosity, default is 2:
            - 0: Print nothing.
            - 1: Print warnings.
            - 2: Print additional runtime and kernel calls
            - 3: Print additional start/stop, condensed parameter information.
            - 4: Print additional full parameter information


    Returns
    -------
    EM : EMArray, (nfreqtime, nrec, nsrc)
        Frequency- or time-domain EM field (depending on ``signal``):
            - If rec is electric, returns E [V/m].
            - If rec is magnetic, returns H [A/m].

        EMArray is a subclassed ndarray with ``.pha`` and ``.amp`` attributes
        (only relevant for frequency-domain data).

        The shape of EM is (nfreqtime, nrec, nsrc). However, single dimensions
        are removed.


    Examples
    --------
    >>> import numpy as np
    >>> from empymod import dipole
    >>> src = [0, 0, 100]
    >>> rec = [np.arange(1, 11)*500, np.zeros(10), 200]
    >>> depth = [0, 300, 1000, 1050]
    >>> res = [1e20, .3, 1, 50, 1]
    >>> EMfield = dipole(src, rec, depth, res, freqtime=1, verb=0)
    >>> print(EMfield)
    [  1.68809346e-10 -3.08303130e-10j  -8.77189179e-12 -3.76920235e-11j
      -3.46654704e-12 -4.87133683e-12j  -3.60159726e-13 -1.12434417e-12j
       1.87807271e-13 -6.21669759e-13j   1.97200208e-13 -4.38210489e-13j
       1.44134842e-13 -3.17505260e-13j   9.92770406e-14 -2.33950871e-13j
       6.75287598e-14 -1.74922886e-13j   4.62724887e-14 -1.32266600e-13j]

    """

    # === 1.  LET'S START ============
    t0 = printstartfinish(verb)

    # === 2.  CHECK INPUT ============

    # Backwards compatibility
    htarg, opt = spline_backwards_hankel(ht, htarg, opt)

    # Check times and Fourier Transform arguments, get required frequencies
    # (freq = freqtime if ``signal=None``)
    if signal is not None:
        time, freq, ft, ftarg = check_time(freqtime, signal, ft, ftarg, verb)
    else:
        freq = freqtime

    # Check layer parameters
    model = check_model(depth, res, aniso, epermH, epermV, mpermH, mpermV,
                        xdirect, verb)
    depth, res, aniso, epermH, epermV, mpermH, mpermV, isfullspace = model

    # Check frequency => get etaH, etaV, zetaH, and zetaV
    frequency = check_frequency(freq, res, aniso, epermH, epermV, mpermH,
                                mpermV, verb)
    freq, etaH, etaV, zetaH, zetaV = frequency

    # Update etaH/etaV and zetaH/zetaV according to user-provided model
    if isinstance(res, dict) and 'func_eta' in res:
        etaH, etaV = res['func_eta'](res, locals())
    if isinstance(res, dict) and 'func_zeta' in res:
        zetaH, zetaV = res['func_zeta'](res, locals())

    # Check Hankel transform parameters
    ht, htarg = check_hankel(ht, htarg, verb)

    # Check optimization
    use_ne_eval, loop_freq, loop_off = check_opt(opt, loop, ht, htarg, verb)

    # Check src-rec configuration
    # => Get flags if src or rec or both are magnetic (msrc, mrec)
    ab_calc, msrc, mrec = check_ab(ab, verb)

    # Check src and rec
    src, nsrc = check_dipole(src, 'src', verb)
    rec, nrec = check_dipole(rec, 'rec', verb)

    # Get offsets and angles (off, angle)
    off, angle = get_off_ang(src, rec, nsrc, nrec, verb)

    # Get layer number in which src and rec reside (lsrc/lrec)
    lsrc, zsrc = get_layer_nr(src, depth)
    lrec, zrec = get_layer_nr(rec, depth)

    # === 3. EM-FIELD CALCULATION ============

    # Collect variables for fem
    inp = (ab_calc, off, angle, zsrc, zrec, lsrc, lrec, depth, freq, etaH,
           etaV, zetaH, zetaV, xdirect, isfullspace, ht, htarg, use_ne_eval,
           msrc, mrec, loop_freq, loop_off)
    EM, kcount, conv = fem(*inp)

    # In case of QWE/QUAD, print Warning if not converged
    conv_warning(conv, htarg, 'Hankel', verb)

    # Do f->t transform if required
    if signal is not None:
        EM, conv = tem(EM, off, freq, time, signal, ft, ftarg)

        # In case of QWE/QUAD, print Warning if not converged
        conv_warning(conv, ftarg, 'Fourier', verb)

    # Reshape for number of sources
    EM = np.squeeze(EM.reshape((-1, nrec, nsrc), order='F'))

    # === 4.  FINISHED ============
    printstartfinish(verb, t0, kcount)

    return EMArray(EM)


def loop(src, rec, depth, res, freqtime, signal=None, aniso=None, epermH=None,
         epermV=None, mpermH=None, mpermV=None, mrec=True, recpts=1,
         strength=0, xdirect=False, ht='fht', htarg=None, ft='sin', ftarg=None,
         opt=None, loop=None, verb=2):
    r"""Return EM fields due to a magnetic source loop.

    Calculate the electromagnetic frequency- or time-domain field due to
    an arbitrary rotated, magnetic source consisting of an electric loop,
    measured by arbitrary rotated, finite electric or magnetic bipole
    receivers or arbitrary rotated magnetic receivers consisting of electric
    loops. By default, the electromagnetic response is normalized to source
    loop area of 1 m2 and receiver length or area of 1 m or 1 m2, respectively,
    and source strength of 1 A.

    A magnetic dipole, as used in ``dipole`` and ``bipole``, has a moment of
    :math:`I^m ds`. However, if the magnetic dipole is generated by an
    electric-wire loop, this changes to :math:`I^m = i\omega\mu A I^e`, where A
    is the area of the loop. The same factor :math:`i\omega\mu A`, applies to
    the receiver, if it consists of an electric-wire loop.

    The current implementation only handles loop sources and receivers in
    layers where :math:`\mu_r^h=\mu_r^v`; the horizontal magnetic permeability
    is used, and a warning is thrown if the vertical differs from the
    horizontal one.

    Note that the kernel internally still calculates dipole sources and
    receivers, the moment is a factor that is multiplied in the frequency
    domain. The logs will therefore still inform about bipoles and dipoles.


    See Also
    --------
    dipole : EM fields due to infinitesimal small EM dipoles.
    bipole : EM fields due to arbitrary rotated, finite length EM dipoles.


    Parameters
    ----------
    src, rec : list of floats or arrays
        Source and receiver coordinates (m):
            - [x0, x1, y0, y1, z0, z1] (bipole of finite length)
            - [x, y, z, azimuth, dip]  (dipole, infinitesimal small)

        Dimensions:
            - The coordinates x, y, and z (dipole) or x0, x1, y0, y1, z0, and
              z1 (bipole) can be single values or arrays.
            - The variables x and y (dipole) or x0, x1, y0, and y1 (bipole)
              must have the same dimensions.
            - The variable z (dipole) or z0 and z1 (bipole) must either be
              single values or having the same dimension as the other
              coordinates.
            - The variables azimuth and dip must be single values. If they have
              different angles, you have to use the bipole-method (with
              recpts = 1, so it is calculated as dipoles). Note that srcpts is
              fixed to 1, as the source is a loop.

        Angles (coordinate system is left-handed, positive z down
        (East-North-Depth):

            - azimuth (°): horizontal deviation from x-axis, anti-clockwise.
            - dip (°): vertical deviation from xy-plane downwards.

        Sources or receivers placed on a layer interface are considered in the
        upper layer.

    depth : list
        Absolute layer interfaces z (m); #depth = #res - 1
        (excluding +/- infinity).

    res : array_like
        Horizontal resistivities rho_h (Ohm.m); #res = #depth + 1.

        Alternatively, res can be a dictionary. See the main manual of empymod
        too see how to exploit this hook to re-calculate etaH, etaV, zetaH, and
        zetaV, which can be used to, for instance, use the Cole-Cole model for
        IP.

    freqtime : array_like
        Frequencies f (Hz) if ``signal`` == None, else times t (s); (f, t > 0).

    signal : {None, 0, 1, -1}, optional
        Source signal, default is None:
            - None: Frequency-domain response
            - -1 : Switch-off time-domain response
            - 0 : Impulse time-domain response
            - +1 : Switch-on time-domain response

    aniso : array_like, optional
        Anisotropies lambda = sqrt(rho_v/rho_h) (-); #aniso = #res.
        Defaults to ones.

    epermH, epermV : array_like, optional
        Relative horizontal/vertical electric permittivities
        epsilon_h/epsilon_v (-);
        #epermH = #epermV = #res. Default is ones.

    mpermH, mpermV : array_like, optional
        Relative horizontal/vertical magnetic permeabilities mu_h/mu_v (-);
        #mpermH = #mpermV = #res. Default is ones.

        Note that the relative horizontal and vertical magnetic permeabilities
        in layers with loop sources or receivers will be set to 1.

    mrec : boolean or string, optional
        Receiver options; default is True:
            - True: Magnetic dipole receiver;
            - False: Electric dipole receiver;
            - 'loop': Magnetic receiver consisting of an electric-wire loop.

    recpts : int, optional
        Number of integration points for bipole receiver, default is 1:
            - recpts < 3  : bipole, but calculated as dipole at centre
            - recpts >= 3 : bipole

        Note that if `mrec='loop'`, `recpts` will be set to 1.

    strength : float, optional
        Source strength (A):
          - If 0, output is normalized to source of 1 m2 area and receiver of 1
            m length or 1 m2 area, and source strength of 1 A.
          - If != 0, output is returned for given source strength and receiver
            length (if `mrec!='loop'`).

        The strength is simply a multiplication factor. It can also be used to
        provide the source and receiver loop area, or also to multiply by
        :math:\mu_0`, if you want the B-field instead of the H-field.

        Default is 0.

    xdirect : bool or None, optional
        Direct field calculation (only if src and rec are in the same layer):
          - If True, direct field is calculated analytically in the frequency
            domain.
          - If False, direct field is calculated in the wavenumber domain.
          - If None, direct field is excluded from the calculation, and only
            reflected fields are returned (secondary field).

        Defaults to False.

    ht : {'fht', 'qwe', 'quad'}, optional
        Flag to choose either the *Digital Linear Filter* method (FHT, *Fast
        Hankel Transform*), the *Quadrature-With-Extrapolation* (QWE), or a
        simple *Quadrature* (QUAD) for the Hankel transform.  Defaults to
        'fht'.

    htarg : dict or list, optional
        Depends on the value for ``ht``:
            - If ``ht`` = 'fht': [fhtfilt, pts_per_dec]:

                - fhtfilt: string of filter name in ``empymod.filters`` or
                           the filter method itself.
                           (default: ``empymod.filters.key_201_2009()``)
                - pts_per_dec: points per decade; (default: 0)
                    - If 0: Standard DLF.
                    - If < 0: Lagged Convolution DLF.
                    - If > 0: Splined DLF

            - If ``ht`` = 'qwe': [rtol, atol, nquad, maxint, pts_per_dec,
                                diff_quad, a, b, limit]:

                - rtol: relative tolerance (default: 1e-12)
                - atol: absolute tolerance (default: 1e-30)
                - nquad: order of Gaussian quadrature (default: 51)
                - maxint: maximum number of partial integral intervals
                          (default: 40)
                - pts_per_dec: points per decade; (default: 0)
                    - If 0, no interpolation is used.
                    - If > 0, interpolation is used.

                - diff_quad: criteria when to swap to QUAD (only relevant if
                  opt='spline') (default: 100)
                - a: lower limit for QUAD (default: first interval from QWE)
                - b: upper limit for QUAD (default: last interval from QWE)
                - limit: limit for quad (default: maxint)

            - If ``ht`` = 'quad': [atol, rtol, limit, lmin, lmax, pts_per_dec]:

                - rtol: relative tolerance (default: 1e-12)
                - atol: absolute tolerance (default: 1e-20)
                - limit: An upper bound on the number of subintervals used in
                  the adaptive algorithm (default: 500)
                - lmin: Minimum wavenumber (default 1e-6)
                - lmax: Maximum wavenumber (default 0.1)
                - pts_per_dec: points per decade (default: 40)

        The values can be provided as dict with the keywords, or as list.
        However, if provided as list, you have to follow the order given above.
        A few examples, assuming ``ht`` = ``qwe``:

            - Only changing rtol:
                {'rtol': 1e-4} or [1e-4] or 1e-4
            - Changing rtol and nquad:
                {'rtol': 1e-4, 'nquad': 101} or [1e-4, '', 101]
            - Only changing diff_quad:
                {'diffquad': 10} or ['', '', '', '', '', 10]

    ft : {'sin', 'cos', 'qwe', 'fftlog', 'fft'}, optional
        Only used if ``signal`` != None. Flag to choose either the Digital
        Linear Filter method (Sine- or Cosine-Filter), the
        Quadrature-With-Extrapolation (QWE), the FFTLog, or the FFT for the
        Fourier transform.  Defaults to 'sin'.

    ftarg : dict or list, optional
        Only used if ``signal`` !=None. Depends on the value for ``ft``:
            - If ``ft`` = 'sin' or 'cos': [fftfilt, pts_per_dec]:

                - fftfilt: string of filter name in ``empymod.filters`` or
                           the filter method itself.
                           (Default: ``empymod.filters.key_201_CosSin_2012()``)
                - pts_per_dec: points per decade; (default: -1)
                    - If 0: Standard DLF.
                    - If < 0: Lagged Convolution DLF.
                    - If > 0: Splined DLF


            - If ``ft`` = 'qwe': [rtol, atol, nquad, maxint, pts_per_dec]:

                - rtol: relative tolerance (default: 1e-8)
                - atol: absolute tolerance (default: 1e-20)
                - nquad: order of Gaussian quadrature (default: 21)
                - maxint: maximum number of partial integral intervals
                          (default: 200)
                - pts_per_dec: points per decade (default: 20)
                - diff_quad: criteria when to swap to QUAD (default: 100)
                - a: lower limit for QUAD (default: first interval from QWE)
                - b: upper limit for QUAD (default: last interval from QWE)
                - limit: limit for quad (default: maxint)

            - If ``ft`` = 'fftlog': [pts_per_dec, add_dec, q]:

                - pts_per_dec: sampels per decade (default: 10)
                - add_dec: additional decades [left, right] (default: [-2, 1])
                - q: exponent of power law bias (default: 0); -1 <= q <= 1

            - If ``ft`` = 'fft': [dfreq, nfreq, ntot]:

                - dfreq: Linear step-size of frequencies (default: 0.002)
                - nfreq: Number of frequencies (default: 2048)
                - ntot:  Total number for FFT; difference between nfreq and
                         ntot is padded with zeroes. This number is ideally a
                         power of 2, e.g. 2048 or 4096 (default: nfreq).
                - pts_per_dec : points per decade (default: None)

                Padding can sometimes improve the result, not always. The
                default samples from 0.002 Hz - 4.096 Hz. If pts_per_dec is set
                to an integer, calculated frequencies are logarithmically
                spaced with the given number per decade, and then interpolated
                to yield the required frequencies for the FFT.

        The values can be provided as dict with the keywords, or as list.
        However, if provided as list, you have to follow the order given above.
        See ``htarg`` for a few examples.

    opt : {None, 'parallel'}, optional
        Optimization flag. Defaults to None:
            - None: Normal case, no parallelization nor interpolation is used.
            - If 'parallel', the package ``numexpr`` is used to evaluate the
              most expensive statements. Always check if it actually improves
              performance for a specific problem. It can speed up the
              calculation for big arrays, but will most likely be slower for
              small arrays. It will use all available cores for these specific
              statements, which all contain ``Gamma`` in one way or another,
              which has dimensions (#frequencies, #offsets, #layers, #lambdas),
              therefore can grow pretty big. The module ``numexpr`` uses by
              default all available cores up to a maximum of 8. You can change
              this behaviour to your desired number of threads ``nthreads``
              with ``numexpr.set_num_threads(nthreads)``.

    loop : {None, 'freq', 'off'}, optional
        Define if to calculate everything vectorized or if to loop over
        frequencies ('freq') or over offsets ('off'), default is None. It
        always loops over frequencies if ``ht = 'qwe'`` or if ``opt =
        'spline'``. Calculating everything vectorized is fast for few offsets
        OR for few frequencies. However, if you calculate many frequencies for
        many offsets, it might be faster to loop over frequencies. Only
        comparing the different versions will yield the answer for your
        specific problem at hand!

    verb : {0, 1, 2, 3, 4}, optional
        Level of verbosity, default is 2:
            - 0: Print nothing.
            - 1: Print warnings.
            - 2: Print additional runtime and kernel calls
            - 3: Print additional start/stop, condensed parameter information.
            - 4: Print additional full parameter information


    Returns
    -------
    EM : EMArray, (nfreqtime, nrec, nsrc)
        Frequency- or time-domain EM field (depending on ``signal``):
            - If rec is electric, returns E [V/m].
            - If rec is magnetic, returns H [A/m].

        EMArray is a subclassed ndarray with ``.pha`` and ``.amp`` attributes
        (only relevant for frequency-domain data).


    Examples
    --------
    >>> import numpy as np
    >>> from empymod import loop
    >>> # z-directed loop source: x, y, z, azimuth, dip
    >>> src = [0, 0, 0, 0, 90]
    >>> # z-directed magnetic dipole receiver-array: x, y, z, azimuth, dip
    >>> rec = [np.arange(1, 11)*500, np.zeros(10), 200, 0, 90]
    >>> # layer boundaries
    >>> depth = [0, 300, 500]
    >>> # layer resistivities
    >>> res = [2e14, 10, 500, 10]
    >>> # Frequency
    >>> freq = 1
    >>> # Calculate magnetic field due to a loop source at 1 Hz.
    >>> # [mrec = True (default)]
    >>> EMfield = loop(src, rec, depth, res, freq, verb=4)
    :: empymod START  ::
    ~
       depth       [m] :  0 300 500
       res     [Ohm.m] :  2E+14 10 500 10
       aniso       [-] :  1 1 1 1
       epermH      [-] :  1 1 1 1
       epermV      [-] :  1 1 1 1
       mpermH      [-] :  1 1 1 1
       mpermV      [-] :  1 1 1 1
       direct field    :  Calc. in wavenumber domain
       frequency  [Hz] :  1
       Hankel          :  DLF (Fast Hankel Transform)
         > Filter      :  Key 201 (2009)
         > DLF type    :  Standard
       Kernel Opt.     :  None
       Loop over       :  None (all vectorized)
       Source(s)       :  1 dipole(s)
         > x       [m] :  0
         > y       [m] :  0
         > z       [m] :  0
         > azimuth [°] :  0
         > dip     [°] :  90
       Receiver(s)     :  10 dipole(s)
         > x       [m] :  500 - 5000 : 10  [min-max; #]
                       :  500 1000 1500 2000 2500 3000 3500 4000 4500 5000
         > y       [m] :  0 - 0 : 10  [min-max; #]
                       :  0 0 0 0 0 0 0 0 0 0
         > z       [m] :  200
         > azimuth [°] :  0
         > dip     [°] :  90
       Required ab's   :  33
    ~
    :: empymod END; runtime = 0:00:00.005025 :: 1 kernel call(s)

    >>> print(EMfield)
    [ -3.05449848e-10 -2.00374185e-11j  -7.12528991e-11 -5.37083268e-12j
      -2.52076501e-11 -1.62732412e-12j  -1.18412295e-11 -8.99570998e-14j
      -6.44054097e-12 +5.61150066e-13j  -3.77109625e-12 +7.89022722e-13j
      -2.28484774e-12 +8.08897623e-13j  -1.40021365e-12 +7.32151174e-13j
      -8.55487532e-13 +6.18402706e-13j  -5.15642408e-13 +4.99091919e-13j]

    """

    # === 1.  LET'S START ============
    t0 = printstartfinish(verb)

    # === 2.  CHECK INPUT ============

    # Check times and Fourier Transform arguments and get required frequencies
    if signal is None:
        freq = freqtime
    else:
        time, freq, ft, ftarg = check_time(freqtime, signal, ft, ftarg, verb)

    # Check layer parameters
    model = check_model(depth, res, aniso, epermH, epermV, mpermH, mpermV,
                        xdirect, verb)
    depth, res, aniso, epermH, epermV, mpermH, mpermV, isfullspace = model

    # Check frequency => get etaH, etaV, zetaH, and zetaV
    frequency = check_frequency(freq, res, aniso, epermH, epermV, mpermH,
                                mpermV, verb)
    freq, etaH, etaV, zetaH, zetaV = frequency

    # Update etaH/etaV and zetaH/zetaV according to user-provided model
    if isinstance(res, dict) and 'func_eta' in res:
        etaH, etaV = res['func_eta'](res, locals())
    if isinstance(res, dict) and 'func_zeta' in res:
        zetaH, zetaV = res['func_zeta'](res, locals())

    # Check Hankel transform parameters
    ht, htarg = check_hankel(ht, htarg, verb)

    # Check optimization
    use_ne_eval, loop_freq, loop_off = check_opt(opt, loop, ht, htarg, verb)

    # Check src and rec, get flags if dipole or not
    # nsrcz/nrecz are number of unique src/rec-pole depths
    src, nsrc, nsrcz, srcdipole = check_bipole(src, 'src')
    rec, nrec, nrecz, recdipole = check_bipole(rec, 'rec')

    # Check if receiver is a loop too.
    if mrec == 'loop':
        rec_loop = True
        mrec = True
        recpts = 1  # If loop, there is no integration.
    else:
        rec_loop = False

    # === 3. EM-FIELD CALCULATION ============

    # Pre-allocate output EM array
    EM = np.zeros((freq.size, nrec*nsrc), dtype=etaH.dtype)

    # Initialize kernel count, conv (only for QWE)
    # (how many times the wavenumber-domain kernel was calld)
    kcount = 0
    conv = True

    # Define some indeces
    isrc = int(nsrc/nsrcz)  # this is either 1 or nsrc
    irec = int(nrec/nrecz)  # this is either 1 or nrec
    isrz = int(isrc*irec)   # this is either 1, nsrc, nrec, or nsrc*nrec

    # The kernel handles only 1 ab with one srcz-recz combination at once.
    # Hence we have to loop over every different depth of src or rec, and
    # over all required ab's.
    for isz in range(nsrcz):  # Loop over source depths

        # Get this source
        srcazmdip = get_azm_dip(src, isz, nsrcz, 1, srcdipole, strength,
                                'src', verb)
        tsrc, srcazm, srcdip, _, _, src_w = srcazmdip

        for irz in range(nrecz):  # Loop over receiver depths

            # Get this receiver
            recazmdip = get_azm_dip(rec, irz, nrecz, recpts, recdipole,
                                    strength, 'rec', verb)
            trec, recazm, recdip, recg_w, recpts, rec_w = recazmdip

            # Get required ab's
            ab_calc = get_abs(True, mrec, srcazm, srcdip, recazm, recdip, verb)

            # Get layer number in which src resides
            lsrc, zsrc = get_layer_nr(tsrc, depth)

            # Check mu at source level.
            if verb > 0 and mpermH[lsrc] != mpermV[lsrc]:
                print('* WARNING :: `mpermH != mpermV` at source level, '
                      'only `mpermH` considered for loop factor.')

            # Pre-allocate temporary receiver EM arrays for integr. loop
            rEM = np.zeros((freq.size, isrz), dtype=etaH.dtype)

            for irg in range(recpts):  # Loop over rec integration pts
                # Note, if source or receiver is a bipole, but horizontal
                # (dip=0), then calculation could be sped up by not looping
                # over the bipole elements, but calculate it all in one go.

                # This integration receiver
                tirec = [trec[0][irg::recpts], trec[1][irg::recpts],
                         trec[2][irg]]

                # Get src-rec offsets and angles
                off, angle = get_off_ang(tsrc, tirec, isrc, irec, verb)

                # Get layer number in which rec resides
                lrec, zrec = get_layer_nr(tirec, depth)

                # Check mu at receiver level.
                if rec_loop and verb > 0 and mpermH[lrec] != mpermV[lrec]:
                    print('* WARNING :: `mpermH != mpermV` at receiver level, '
                          'only `mpermH` considered for loop factor.')

                # Gather variables
                finp = (off, angle, zsrc, zrec, lsrc, lrec, depth, freq,
                        etaH, etaV, zetaH, zetaV, xdirect, isfullspace, ht,
                        htarg, use_ne_eval, True, mrec, loop_freq,
                        loop_off, conv)

                # Pre-allocate temporary EM array for ab-loop
                abEM = np.zeros((freq.size, isrz), dtype=etaH.dtype)

                for iab in ab_calc:  # Loop over required ab's

                    # Carry-out the frequency-domain calculation
                    out = fem(iab, *finp)

                    # Get geometrical scaling factor
                    tfact = get_geo_fact(iab, srcazm, srcdip, recazm, recdip,
                                         True, mrec)

                    # Add field to EM with geometrical factor
                    abEM += out[0]*np.squeeze(tfact)

                    # Update kernel count
                    kcount += out[1]

                    # Update conv (QWE convergence)
                    conv *= out[2]

                # Add this receiver element, with weight from integration
                rEM += abEM*recg_w[irg]

            # Scale signal for src-strength and rec-lengths
            src_rec_w = 1
            if strength > 0:
                src_rec_w *= np.repeat(src_w, irec)
                src_rec_w *= np.tile(rec_w, isrc)
            rEM *= src_rec_w

            # Add this src-rec signal
            if nrec == nrecz:
                if nsrc == nsrcz:  # Case 1: Looped over each src and each rec
                    EM[:, isz*nrec+irz:isz*nrec+irz+1] = rEM
                else:              # Case 2: Looped over each rec
                    EM[:, irz:nsrc*nrec:nrec] = rEM
            else:
                if nsrc == nsrcz:  # Case 3: Looped over each src
                    EM[:, isz*nrec:nrec*(isz+1)] = rEM
                else:              # Case 4: All in one go
                    EM = rEM

    # In case of QWE/QUAD, print Warning if not converged
    conv_warning(conv, htarg, 'Hankel', verb)

    # Multiplication with frequency-dependent loop factors.
    EM *= zetaH[:, lsrc, None]
    if rec_loop:
        EM *= zetaH[:, lrec, None]

    # Do f->t transform if required
    if signal is not None:
        EM, conv = tem(EM, EM[0, :], freq, time, signal, ft, ftarg)

        # In case of QWE/QUAD, print Warning if not converged
        conv_warning(conv, ftarg, 'Fourier', verb)

    # Reshape for number of sources
    EM = np.squeeze(EM.reshape((-1, nrec, nsrc), order='F'))

    # === 4.  FINISHED ============
    printstartfinish(verb, t0, kcount)

    return EMArray(EM)


def analytical(src, rec, res, freqtime, solution='fs', signal=None, ab=11,
               aniso=None, epermH=None, epermV=None, mpermH=None, mpermV=None,
               verb=2):
    r"""Return analytical full- or half-space solution.

    Calculate the electromagnetic frequency- or time-domain field due to
    infinitesimal small electric or magnetic dipole source(s), measured by
    infinitesimal small electric or magnetic dipole receiver(s); sources and
    receivers are directed along the principal directions x, y, or z, and all
    sources are at the same depth, as well as all receivers are at the same
    depth.

    In the case of a halfspace the air-interface is located at z = 0 m.

    You can call the functions ``fullspace`` and ``halfspace`` in ``kernel.py``
    directly. This interface is just to provide a consistent interface with the
    same input parameters as for instance for ``dipole``.

    This function yields the same result if ``solution='fs'`` as ``dipole``, if
    the model is a fullspace.

    Included are:
      - Full fullspace solution (``solution='fs'``) for ee-, me-, em-,
        mm-fields, only frequency domain, [HuTS15]_.
      - Diffusive fullspace solution (``solution='dfs'``) for ee-fields,
        [SlHM10]_.
      - Diffusive halfspace solution (``solution='dhs'``) for ee-fields,
        [SlHM10]_.
      - Diffusive direct- and reflected field and airwave
        (``solution='dsplit'``) for ee-fields, [SlHM10]_.
      - Diffusive direct- and reflected field and airwave
        (``solution='dtetm'``) for ee-fields, split into TE and TM mode
        [SlHM10]_.

    Parameters
    ----------
    src, rec : list of floats or arrays
        Source and receiver coordinates (m): [x, y, z].
        The x- and y-coordinates can be arrays, z is a single value.
        The x- and y-coordinates must have the same dimension.

    res : float
        Horizontal resistivity rho_h (Ohm.m).

        Alternatively, res can be a dictionary. See the main manual of empymod
        too see how to exploit this hook to re-calculate etaH, etaV, zetaH, and
        zetaV, which can be used to, for instance, use the Cole-Cole model for
        IP.

    freqtime : array_like
        Frequencies f (Hz) if ``signal`` == None, else times t (s); (f, t > 0).

    solution : str, optional
      Defines which solution is returned:
        - 'fs' : Full fullspace solution (ee-, me-, em-, mm-fields); f-domain.
        - 'dfs' : Diffusive fullspace solution (ee-fields only).
        - 'dhs' : Diffusive halfspace solution (ee-fields only).
        - 'dsplit' : Diffusive direct- and reflected field and airwave
                     (ee-fields only).
        - 'dtetm' : as dsplit, but direct fielt TE, TM; reflected field TE, TM,
                    and airwave (ee-fields only).

    signal : {None, 0, 1, -1}, optional
        Source signal, default is None:
            - None: Frequency-domain response
            - -1 : Switch-off time-domain response
            - 0 : Impulse time-domain response
            - +1 : Switch-on time-domain response

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

    aniso : float, optional
        Anisotropy lambda = sqrt(rho_v/rho_h) (-); defaults to one.

    epermH, epermV : float, optional
        Relative horizontal/vertical electric permittivity epsilon_h/epsilon_v
        (-); default is one. Ignored for the diffusive solution.

    mpermH, mpermV : float, optional
        Relative horizontal/vertical magnetic permeability mu_h/mu_v (-);
        default is one. Ignored for the diffusive solution.

    verb : {0, 1, 2, 3, 4}, optional
        Level of verbosity, default is 2:
            - 0: Print nothing.
            - 1: Print warnings.
            - 2: Print additional runtime
            - 3: Print additional start/stop, condensed parameter information.
            - 4: Print additional full parameter information

    Returns
    -------
    EM : EMArray, (nfreqtime, nrec, nsrc)
        Frequency- or time-domain EM field (depending on ``signal``):
            - If rec is electric, returns E [V/m].
            - If rec is magnetic, returns H [A/m].

        EMArray is a subclassed ndarray with ``.pha`` and ``.amp`` attributes
        (only relevant for frequency-domain data).

        The shape of EM is (nfreqtime, nrec, nsrc). However, single dimensions
        are removed.

        If ``solution='dsplit'``, three ndarrays are returned: direct, reflect,
        air.

        If ``solution='dtetm'``, five ndarrays are returned: direct_TE,
        direct_TM, reflect_TE, reflect_TM, air.


    Examples
    --------
    >>> import numpy as np
    >>> from empymod import analytical
    >>> src = [0, 0, 0]
    >>> rec = [np.arange(1, 11)*500, np.zeros(10), 200]
    >>> res = 50
    >>> EMfield = analytical(src, rec, res, freqtime=1, verb=0)
    >>> print(EMfield)
    [  4.03091405e-08 -9.69163818e-10j   6.97630362e-09 -4.88342150e-10j
       2.15205979e-09 -2.97489809e-10j   8.90394459e-10 -1.99313433e-10j
       4.32915802e-10 -1.40741644e-10j   2.31674165e-10 -1.02579391e-10j
       1.31469130e-10 -7.62770461e-11j   7.72342470e-11 -5.74534125e-11j
       4.61480481e-11 -4.36275540e-11j   2.76174038e-11 -3.32860932e-11j]
    """

    # === 1.  LET'S START ============
    t0 = printstartfinish(verb)

    # === 2.  CHECK INPUT ============

    # Check times or frequencies
    if signal is not None:
        freqtime = check_time_only(freqtime, signal, verb)

    # Check layer parameters
    model = check_model([], res, aniso, epermH, epermV, mpermH, mpermV, True,
                        verb)
    depth, res, aniso, epermH, epermV, mpermH, mpermV, _ = model

    # Check frequency => get etaH, etaV, zetaH, and zetaV
    frequency = check_frequency(freqtime, res, aniso, epermH, epermV, mpermH,
                                mpermV, verb)
    freq, etaH, etaV, zetaH, zetaV = frequency

    # Update etaH/etaV and zetaH/zetaV according to user-provided model
    if isinstance(res, dict) and 'func_eta' in res:
        etaH, etaV = res['func_eta'](res, locals())
    if isinstance(res, dict) and 'func_zeta' in res:
        zetaH, zetaV = res['func_zeta'](res, locals())

    # Check src-rec configuration
    # => Get flags if src or rec or both are magnetic (msrc, mrec)
    ab_calc, msrc, mrec = check_ab(ab, verb)

    # Check src and rec
    src, nsrc = check_dipole(src, 'src', verb)
    rec, nrec = check_dipole(rec, 'rec', verb)

    # Get offsets and angles (off, angle)
    off, angle = get_off_ang(src, rec, nsrc, nrec, verb)

    # Get layer number in which src and rec reside (lsrc/lrec)
    _, zsrc = get_layer_nr(src, depth)
    _, zrec = get_layer_nr(rec, depth)

    # Check possibilities
    check_solution(solution, signal, ab, msrc, mrec)

    # === 3. EM-FIELD CALCULATION ============

    if solution[0] == 'd':
        # To make it work with Laplace domain calculations.
        if signal is None:
            if np.any(freqtime > 0):
                freqtime = 2j*np.pi*freq
            else:
                freqtime = freq
        EM = kernel.halfspace(off, angle, zsrc, zrec, etaH, etaV,
                              freqtime[:, None], ab_calc, signal, solution)
    else:
        if ab_calc not in [36, ]:
            EM = kernel.fullspace(off, angle, zsrc, zrec, etaH, etaV, zetaH,
                                  zetaV, ab_calc, msrc, mrec)
        else:
            # If <ab> = 36 (or 63), field is zero
            # In `bipole` and in `dipole`, this is taken care of in `fem`. Here
            # we have to take care of it separately
            EM = np.zeros((freq.size*nrec*nsrc), dtype=etaH.dtype)

    # Squeeze
    if solution[1:] == 'split':
        EM = (np.squeeze(EM[0].reshape((-1, nrec, nsrc), order='F')),
              np.squeeze(EM[1].reshape((-1, nrec, nsrc), order='F')),
              np.squeeze(EM[2].reshape((-1, nrec, nsrc), order='F')))
    elif solution[1:] == 'tetm':
        EM = (np.squeeze(EM[0].reshape((-1, nrec, nsrc), order='F')),
              np.squeeze(EM[1].reshape((-1, nrec, nsrc), order='F')),
              np.squeeze(EM[2].reshape((-1, nrec, nsrc), order='F')),
              np.squeeze(EM[3].reshape((-1, nrec, nsrc), order='F')),
              np.squeeze(EM[4].reshape((-1, nrec, nsrc), order='F')))
    else:
        EM = np.squeeze(EM.reshape((-1, nrec, nsrc), order='F'))

    # === 4.  FINISHED ============
    printstartfinish(verb, t0)

    return EMArray(EM)


def gpr(src, rec, depth, res, freqtime, cf, gain=None, ab=11, aniso=None,
        epermH=None, epermV=None, mpermH=None, mpermV=None, xdirect=False,
        ht='quad', htarg=None, ft='fft', ftarg=None, opt=None, loop=None,
        verb=2):
    r"""Return Ground-Penetrating Radar signal.

    THIS FUNCTION IS EXPERIMENTAL, USE WITH CAUTION.

    It is rather an example how you can calculate GPR responses; however, DO
    NOT RELY ON IT! It works only well with QUAD or QWE (``quad``, ``qwe``) for
    the Hankel transform, and with FFT (``fft``) for the Fourier transform.

    It calls internally ``dipole`` for the frequency-domain calculation. It
    subsequently convolves the response with a Ricker wavelet with central
    frequency ``cf``. If signal!=None, it carries out the Fourier transform and
    applies a gain to the response.

    For input parameters see the function ``dipole``, except for:

    Parameters
    ----------
    cf : float
        Centre frequency of GPR-signal, in Hz. Sensible values are between
        10 MHz and 3000 MHz.

    gain : float
        Power of gain function. If None, no gain is applied. Only used if
        signal!=None.


    Returns
    -------
    EM : ndarray
        GPR response

    """
    if verb > 2:
        print("   GPR             :  EXPERIMENTAL, USE WITH CAUTION")
        print("     > centre freq :  " + str(cf))
        print("     > gain        :  " + str(gain))

    # === 1.  CHECK TIME ============

    # Check times and Fourier Transform arguments, get required frequencies
    time, freq, ft, ftarg = check_time(freqtime, 0, ft, ftarg, verb)

    # === 2. CALL DIPOLE ============

    EM = dipole(src, rec, depth, res, freq, None, ab, aniso, epermH, epermV,
                mpermH, mpermV, xdirect, ht, htarg, ft, ftarg, opt, loop, verb)

    # === 3. GPR STUFF

    # Get required parameters
    src, nsrc = check_dipole(src, 'src', 0)
    rec, nrec = check_dipole(rec, 'rec', 0)
    off, _ = get_off_ang(src, rec, nsrc, nrec, 0)

    # Reshape output from dipole
    EM = EM.reshape((-1, nrec*nsrc), order='F')

    # Multiply with ricker wavelet
    cfc = -(np.r_[0, freq[:-1]]/cf)**2
    fwave = cfc*np.exp(cfc)
    EM *= fwave[:, None]

    # Do f->t transform
    EM, conv = tem(EM, off, freq, time, 0, ft, ftarg)

    # In case of QWE/QUAD, print Warning if not converged
    conv_warning(conv, ftarg, 'Fourier', verb)

    # Apply gain; make pure real
    EM *= (1 + np.abs((time*10**9)**gain))[:, None]
    EM = EM.real

    # Reshape for number of sources
    EM = np.squeeze(EM.reshape((-1, nrec, nsrc), order='F'))

    return EM


def dipole_k(src, rec, depth, res, freq, wavenumber, ab=11, aniso=None,
             epermH=None, epermV=None, mpermH=None, mpermV=None, verb=2):
    r"""Return electromagnetic wavenumber-domain field.

    Calculate the electromagnetic wavenumber-domain field due to infinitesimal
    small electric or magnetic dipole source(s), measured by infinitesimal
    small electric or magnetic dipole receiver(s); sources and receivers are
    directed along the principal directions x, y, or z, and all sources are at
    the same depth, as well as all receivers are at the same depth.


    See Also
    --------
    dipole : EM fields due to infinitesimal small EM dipoles.
    bipole : EM fields due to arbitrary rotated, finite length EM dipoles.
    loop : EM fields due to a magnetic source loop.


    Parameters
    ----------
    src, rec : list of floats or arrays
        Source and receiver coordinates (m): [x, y, z].
        The x- and y-coordinates can be arrays, z is a single value.
        The x- and y-coordinates must have the same dimension.
        The x- and y-coordinates only matter for the angle-dependent factor.

        Sources or receivers placed on a layer interface are considered in the
        upper layer.

    depth : list
        Absolute layer interfaces z (m); #depth = #res - 1
        (excluding +/- infinity).

    res : array_like
        Horizontal resistivities rho_h (Ohm.m); #res = #depth + 1.

    freq : array_like
        Frequencies f (Hz), used to calculate etaH/V and zetaH/V.

    wavenumber : array
        Wavenumbers lambda (1/m)

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

    epermH, epermV : array_like, optional
        Relative horizontal/vertical electric permittivities
        epsilon_h/epsilon_v (-);
        #epermH = #epermV = #res. Default is ones.

    mpermH, mpermV : array_like, optional
        Relative horizontal/vertical magnetic permeabilities mu_h/mu_v (-);
        #mpermH = #mpermV = #res. Default is ones.

    verb : {0, 1, 2, 3, 4}, optional
        Level of verbosity, default is 2:
            - 0: Print nothing.
            - 1: Print warnings.
            - 2: Print additional runtime and kernel calls
            - 3: Print additional start/stop, condensed parameter information.
            - 4: Print additional full parameter information


    Returns
    -------
    PJ0, PJ1 : array
        Wavenumber-domain EM responses:
            - PJ0: Wavenumber-domain solution for the kernel with a Bessel
              function of the first kind of order zero.
            - PJ1: Wavenumber-domain solution for the kernel with a Bessel
              function of the first kind of order one.


    Examples
    --------
    >>> import numpy as np
    >>> from empymod.model import dipole_k
    >>> src = [0, 0, 100]
    >>> rec = [5000, 0, 200]
    >>> depth = [0, 300, 1000, 1050]
    >>> res = [1e20, .3, 1, 50, 1]
    >>> freq = 1
    >>> wavenrs = np.logspace(-3.7, -3.6, 10)
    >>> PJ0, PJ1 = dipole_k(src, rec, depth, res, freq, wavenrs, verb=0)
    >>> print(PJ0)
    [ -1.02638329e-08 +4.91531529e-09j  -1.05289724e-08 +5.04222413e-09j
      -1.08009148e-08 +5.17238608e-09j  -1.10798310e-08 +5.30588284e-09j
      -1.13658957e-08 +5.44279805e-09j  -1.16592877e-08 +5.58321732e-09j
      -1.19601897e-08 +5.72722830e-09j  -1.22687889e-08 +5.87492067e-09j
      -1.25852765e-08 +6.02638626e-09j  -1.29098481e-08 +6.18171904e-09j]
    >>> print(PJ1)
    [  1.79483705e-10 -6.59235332e-10j   1.88672497e-10 -6.93749344e-10j
       1.98325814e-10 -7.30068377e-10j   2.08466693e-10 -7.68286748e-10j
       2.19119282e-10 -8.08503709e-10j   2.30308887e-10 -8.50823701e-10j
       2.42062030e-10 -8.95356636e-10j   2.54406501e-10 -9.42218177e-10j
       2.67371420e-10 -9.91530051e-10j   2.80987292e-10 -1.04342036e-09j]
    """

    # === 1.  LET'S START ============
    t0 = printstartfinish(verb)

    # === 2.  CHECK INPUT ============

    # Check layer parameters (isfullspace not required)
    modl = check_model(depth, res, aniso, epermH, epermV, mpermH, mpermV,
                       False, verb)
    depth, res, aniso, epermH, epermV, mpermH, mpermV, _ = modl

    # Check frequency => get etaH, etaV, zetaH, and zetaV
    f = check_frequency(freq, res, aniso, epermH, epermV, mpermH, mpermV, verb)
    freq, etaH, etaV, zetaH, zetaV = f

    # Check src-rec configuration
    # => Get flags if src or rec or both are magnetic (msrc, mrec)
    ab_calc, msrc, mrec = check_ab(ab, verb)

    # Check src and rec
    src, nsrc = check_dipole(src, 'src', verb)
    rec, nrec = check_dipole(rec, 'rec', verb)

    # Get angle-dependent factor
    off, angle = get_off_ang(src, rec, nsrc, nrec, verb)
    factAng = kernel.angle_factor(angle, ab, msrc, mrec)

    # Get layer number in which src and rec reside (lsrc/lrec)
    lsrc, zsrc = get_layer_nr(src, depth)
    lrec, zrec = get_layer_nr(rec, depth)

    # === 3. EM-FIELD CALCULATION ============

    # Pre-allocate
    if off.size == 1 and np.ndim(wavenumber) == 2:
        PJ0 = np.zeros((freq.size, wavenumber.shape[0], wavenumber.shape[1]),
                       dtype=etaH.dtype)
        PJ1 = np.zeros((freq.size, wavenumber.shape[0], wavenumber.shape[1]),
                       dtype=etaH.dtype)
    else:
        PJ0 = np.zeros((freq.size, off.size, wavenumber.size),
                       dtype=etaH.dtype)
        PJ1 = np.zeros((freq.size, off.size, wavenumber.size),
                       dtype=etaH.dtype)

    # If <ab> = 36 (or 63), field is zero
    # In `bipole` and in `dipole`, this is taken care of in `fem`. Here we
    # have to take care of it separately
    if ab_calc not in [36, ]:

        # Calculate wavenumber response
        J0, J1, J0b = kernel.wavenumber(zsrc, zrec, lsrc, lrec, depth, etaH,
                                        etaV, zetaH, zetaV,
                                        np.atleast_2d(wavenumber), ab_calc,
                                        False, msrc, mrec, False)

        # Collect output
        if J1 is not None:
            PJ1 += factAng[:, np.newaxis]*J1
            if ab in [11, 12, 21, 22, 14, 24, 15, 25]:  # Because of J2
                # J2(kr) = 2/(kr)*J1(kr) - J0(kr)
                PJ1 /= off[:, None]
        if J0 is not None:
            PJ0 += J0
        if J0b is not None:
            PJ0 += factAng[:, np.newaxis]*J0b

    # === 4.  FINISHED ============
    printstartfinish(verb, t0, 1)

    return np.squeeze(PJ0), np.squeeze(PJ1)


def wavenumber(src, rec, depth, res, freq, wavenumber, ab=11, aniso=None,
               epermH=None, epermV=None, mpermH=None, mpermV=None, verb=2):
    r"""Depreciated. Use `dipole_k` instead."""

    # Issue warning
    mesg = ("\n    The use of `model.wavenumber` is deprecated and will " +
            "be removed;\n    use `model.dipole_k` instead.")
    warnings.warn(mesg, DeprecationWarning)

    return dipole_k(src, rec, depth, res, freq, wavenumber, ab, aniso, epermH,
                    epermV, mpermH, mpermV, verb)


# Core modelling routines

def fem(ab, off, angle, zsrc, zrec, lsrc, lrec, depth, freq, etaH, etaV, zetaH,
        zetaV, xdirect, isfullspace, ht, htarg, use_ne_eval, msrc, mrec,
        loop_freq, loop_off, conv=True):
    r"""Return electromagnetic frequency-domain response.

    This function is called from one of the above modelling routines. No
    input-check is carried out here. See the main description of :mod:`model`
    for information regarding input and output parameters.

    This function can be directly used if you are sure the provided input is in
    the correct format. This is useful for inversion routines and similar, as
    it can speed-up the calculation by omitting input-checks.

    """
    # Preallocate array
    fEM = np.zeros((freq.size, off.size), dtype=etaH.dtype)

    # Initialize kernel count
    # (how many times the wavenumber-domain kernel was calld)
    kcount = 0

    # If <ab> = 36 (or 63), fEM-field is zero
    if ab in [36, ]:
        return fEM, kcount, conv

    # Get full-space-solution if xdirect=True and model is a full-space or
    # if src and rec are in the same layer.
    if xdirect and (isfullspace or lsrc == lrec):
        fEM += kernel.fullspace(off, angle, zsrc, zrec, etaH[:, lrec],
                                etaV[:, lrec], zetaH[:, lrec], zetaV[:, lrec],
                                ab, msrc, mrec)

    # If `xdirect = None` we set it here to True, so it is NOT calculated in
    # the wavenumber domain. (Only reflected fields are returned.)
    if xdirect is None:
        xdir = True
    else:
        xdir = xdirect

    # If not full-space with xdirect calculate fEM-field
    if not isfullspace*xdir:

        # Get angle dependent factors
        factAng = kernel.angle_factor(angle, ab, msrc, mrec)

        # Compute required lambdas for given hankel-filter-base
        # This should be in utils, but this is a backwards-incompatible change.
        # Move this to utils for version 2.0.
        if ht == 'fht':
            # htarg[0] = filter; htarg[1] = pts_per_dec
            lambd, int_pts = transform.get_spline_values(
                    htarg[0], off, htarg[1])
            if not loop_off:
                htarg = (htarg[0], htarg[1], lambd, int_pts)

        calc = getattr(transform, ht)
        if loop_freq:

            for i in range(freq.size):
                out = calc(zsrc, zrec, lsrc, lrec, off, factAng, depth, ab,
                           etaH[None, i, :], etaV[None, i, :],
                           zetaH[None, i, :], zetaV[None, i, :], xdir,
                           htarg, use_ne_eval, msrc, mrec)
                fEM[None, i, :] += out[0]
                kcount += out[1]
                conv *= out[2]

        elif loop_off:
            for i in range(off.size):

                # See comments above where it says "ht == 'fht'".
                # Get pre-calculated lambd, int_pts for this offset
                if ht == 'fht':
                    htarg = (htarg[0], htarg[1], lambd[None, i, :], int_pts[i])

                out = calc(zsrc, zrec, lsrc, lrec, off[None, i],
                           factAng[None, i], depth, ab, etaH, etaV, zetaH,
                           zetaV, xdir, htarg, use_ne_eval, msrc, mrec)
                fEM[:, None, i] += out[0]
                kcount += out[1]
                conv *= out[2]
        else:
            out = calc(zsrc, zrec, lsrc, lrec, off, factAng, depth, ab, etaH,
                       etaV, zetaH, zetaV, xdir, htarg, use_ne_eval, msrc,
                       mrec)
            fEM += out[0]
            kcount += out[1]
            conv *= out[2]

    return fEM, kcount, conv


def tem(fEM, off, freq, time, signal, ft, ftarg, conv=True):
    r"""Return time-domain response of the frequency-domain response fEM.

    This function is called from one of the above modelling routines. No
    input-check is carried out here. See the main description of :mod:`model`
    for information regarding input and output parameters.

    This function can be directly used if you are sure the provided input is in
    the correct format. This is useful for inversion routines and similar, as
    it can speed-up the calculation by omitting input-checks.

    """
    # 1. Scale frequencies if switch-on/off response
    # Step function for causal times is like a unit fct, therefore an impulse
    # in frequency domain
    if signal in [-1, 1]:
        # Divide by signal/(2j*pi*f) to obtain step response
        fact = signal/(2j*np.pi*freq)
    else:
        fact = 1

    # 2. f->t transform
    tEM = np.zeros((time.size, off.size))
    for i in range(off.size):
        out = getattr(transform, ft)(fEM[:, i]*fact, time, freq, ftarg)
        tEM[:, i] += out[0]
        conv *= out[1]

    return tEM*2/np.pi, conv  # Scaling from Fourier transform
