"""

:mod:`model` -- Model EM-responses
==================================

EM-modelling routines. The implemented routines might not be the fastest
solution to your specific problem. Use these routines as template to create
your own, problem-specific modelling routine!

Principal routines:
    - `bipole`
    - `dipole`

The main routine is `bipole`, which can model bipole source(s) and bipole
receiver(s) of arbitrary direction, for electric or magnetic sources and
receivers, both in frequency and in time. A subset of `bipole` is `dipole`,
which models infinitesimal small dipoles along the principal axes x, y, and z.

These principal routines make use of the following two core routines:
    - `fem`: Calculate wavenumber-domain electromagnetic field and carry out
             the Hankel transform to the frequency domain.
    - `tem`: Carry out the Fourier transform to time domain after `fem`.

Two further routines are shortcuts for frequency- and time-domain dipoles,
respectively, and mainly in for legacy reasons:

    - `frequency`: Shortcut of `dipole` for frequency-domain calculation.
    - `time`: Shortcut of `dipole` for time-domain calculation.

Two more routines are more kind of examples and cannot be regarded stable;
they can serve as template to create your own routines:

    - `gpr`:        Calculate the Ground-Penetrating Radar (GPR) response.
    - `wavenumber`: Calculate the electromagnetic wavenumber-domain solution.

"""
# Copyright 2016-2017 Dieter Werthmüller
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

from . import kernel, transform
from .utils import (check_time, check_model, check_frequency, check_hankel,
                    check_opt, check_dipole, check_bipole, check_ab, get_abs,
                    get_geo_fact, get_azm_dip, get_off_ang, get_layer_nr,
                    printstartfinish, conv_warning)

__all__ = ['bipole', 'dipole', 'frequency', 'time', 'gpr', 'wavenumber', 'fem',
           'tem']


def bipole(src, rec, depth, res, freqtime, signal=None, aniso=None,
           epermH=None, epermV=None, mpermH=None, mpermV=None, msrc=False,
           srcpts=1, mrec=False, recpts=1, strength=0, xdirect=True,
           ht='fht', htarg=None, ft='sin', ftarg=None, opt=None, loop=None,
           verb=2):
    """Return the electromagnetic field due to an electromagnetic source.

    Calculate the electromagnetic frequency- or time-domain field due to
    arbitrary finite electric or magnetic bipole sources, measured by arbitrary
    finite electric or magnetic bipole receivers. By default, the
    electromagnetic response is normalized to to source and receiver of 1 m
    length, and source strength of 1 A.


    See Also
    --------
    fem : Electromagnetic frequency-domain response.
    tem : Electromagnetic time-domain response.


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

    depth : list
        Absolute layer interfaces z (m); #depth = #res - 1
        (excluding +/- infinity).

    res : array_like
        Horizontal resistivities rho_h (Ohm.m); #res = #depth + 1.

    freqtime : array_like
        Frequencies f (Hz) if `signal` == None, else times t (s).

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
                          (default: `empymod.filters.key_201_2009()`)
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

    ft : {'sin', 'cos', 'qwe', 'fftlog'}, optional
        Only used if `signal` != None. Flag to choose either the Sine- or
        Cosine-Filter, the Quadrature-With-Extrapolation (QWE), or FFTLog for
        the Fourier transform.  Defaults to 'sin'.

    ftarg : str or filter from empymod.filters or array_like, optional
        Only used if `signal` !=None. Depends on the value for `ft`:
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
              The module `numexpr` uses by default all available cores up to a
              maximum of 8. You can change this behaviour to your desired
              number of threads `nthreads` with
              `numexpr.set_num_threads(nthreads)`.
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

    verb : {0, 1, 2, 3, 4}, optional
        Level of verbosity, default is 2:
            - 0: Print nothing.
            - 1: Print warnings.
            - 2: Print additional runtime and kernel calls
            - 3: Print additional start/stop, condensed parameter information.
            - 4: Print additional full parameter information


    Returns
    -------
    EM : ndarray, (nfreq, nrec, nsrc)
        Frequency- or time-domain EM field (depending on `signal`):
            - If rec is electric, returns E [V/m].
            - If rec is magnetic, returns B [T] (not H [A/m]!).

        In the case of the impulse time-domain response, the unit is further
        divided by seconds [1/s].

        However, source and receiver are normalised (unless strength != 0). So
        for instance in the electric case the source strength is 1 A and its
        length is 1 m. So the electric field could also be written as
        [V/(A.m2)].

        The shape of EM is (nfreq, nrec, nsrc). However, single dimensions
        are removed.


    Examples
    --------
    >>> import numpy as np
    >>> from empymod import bipole
    >>> # x-directed bipole source: x0, x1, y0, y1, z0, z1
    >>> src = [-50, 50, 0, 0, 100, 100]
    >>> # x-directed dipole source-array: x, y, z, azimuth, dip
    >>> rec = [np.arange(1, 11)*500, np.zeros(10), 200, 0, 0]
    >>> # layer boundaries
    >>> depth = [0, 300, 1000, 1050]
    >>> # layer resistivities
    >>> res = [1e20, .3, 1, 50, 1]
    >>> # Frequency
    >>> freq = 1
    >>> # Calculate electric field due to an electric source at 1 Hz.
    >>> # [msrc = mrec = True (default)]
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
       Hankel          :  Fast Hankel Transform
         > Filter      :  Key 201 (2009)
       Hankel Opt.     :  None
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
         > azimuth [°] :  0 - 0 : 10  [min-max; #]
                       :  0 0 0 0 0 0 0 0 0 0
         > dip     [°] :  0 - 0 : 10  [min-max; #]
                       :  0 0 0 0 0 0 0 0 0 0
       Required ab's   :  11
    ~
    :: empymod END; runtime = 0:00:00.022349 :: 1 kernel call(s)
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

    # Check times and Fourier Transform arguments and get required frequencies
    if signal is None:
        freq = freqtime
    else:
        time, freq, ft, ftarg = check_time(freqtime, signal, ft, ftarg, verb)

    # Check layer parameters
    model = check_model(depth, res, aniso, epermH, epermV, mpermH, mpermV,
                        verb)
    depth, res, aniso, epermH, epermV, mpermH, mpermV, isfullspace = model

    # Check frequency => get etaH, etaV, zetaH, and zetaV
    frequency = check_frequency(freq, res, aniso, epermH, epermV, mpermH,
                                mpermV, verb)
    freq, etaH, etaV, zetaH, zetaV = frequency

    # Check Hankel transform parameters
    ht, htarg = check_hankel(ht, htarg, verb)

    # Check optimization
    optimization = check_opt(opt, loop, ht, htarg, verb)
    use_spline, use_ne_eval, loop_freq, loop_off = optimization

    # Check src and rec, get flags if dipole or not
    # nsrcz/nrecz are number of unique src/rec-pole depths
    src, nsrc, nsrcz, srcdipole = check_bipole(src, 'src')
    rec, nrec, nrecz, recdipole = check_bipole(rec, 'rec')

    # === 3. EM-FIELD CALCULATION ============

    # Pre-allocate output EM array
    EM = np.zeros((freq.size, nrec*nsrc), dtype=complex)

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
            sEM = np.zeros((freq.size, isrz), dtype=complex)

            for isg in range(srcpts):  # Loop over src integration points

                # This integration source
                tisrc = [tsrc[0][isg::srcpts], tsrc[1][isg::srcpts],
                         tsrc[2][isg]]

                # Get layer number in which src resides
                lsrc, zsrc = get_layer_nr(tisrc, depth)

                # Pre-allocate temporary receiver EM arrays for integr. loop
                rEM = np.zeros((freq.size, isrz), dtype=complex)

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
                            htarg, use_spline, use_ne_eval, msrc, mrec,
                            loop_freq, loop_off, conv)

                    # Pre-allocate temporary EM array for ab-loop
                    abEM = np.zeros((freq.size, isrz), dtype=complex)

                    for iab in ab_calc:  # Loop over required ab's

                        # Carry-out the frequency-domain calculation
                        out = fem(iab, *finp)

                        # Get geometrical scaling factor
                        tfact = get_geo_fact(iab, srcazm, srcdip, recazm,
                                             recdip, msrc, mrec)

                        # Add field to EM with geometrical factor
                        abEM += out[0]*tfact

                        # Update kernel count
                        kcount += out[1]

                        # Update conv (QWE convergence)
                        conv *= out[2]

                    # Add this receiver element, with weight from integration
                    rEM += abEM*recg_w[irg]

                # Add this source element, with weight from integration
                sEM += rEM*srcg_w[isg]

            # Get required indices
            if nrec == nrecz:
                if nsrc == nsrcz:  # Case 1: Looped over each src and each rec
                    si, ei, st = isz*irec + irz, isz*irec + irz + 1, 1
                else:              # Case 2: Looped over each rec
                    si, ei, st = irz, nsrc*nrec, nrec
            else:
                if nsrc == nsrcz:  # Case 3: Looped over each src
                    si, ei, st = isz*irec, (isz+1)*irec, 1
                else:              # Case 4: All in one go
                    si, ei, st = 0, nsrc*nrec, 1

            # Get required scaling from src-strength and src/rec-length
            src_rec_w = 1
            if strength > 0:
                src_rec_w *= np.repeat(src_w, irec)
                src_rec_w *= np.tile(rec_w, isrc)

            # Add this src-rec signal
            EM[:, si:ei:st] = sEM*src_rec_w

    # In case of QWE, print Warning if not converged
    conv_warning(conv, htarg, 'Hankel', verb)

    # Do f->t transform if required
    if signal is not None:
        EM, conv = tem(EM, off, freq, time, signal, ft, ftarg)

        # In case of QWE, print Warning if not converged
        conv_warning(conv, ftarg, 'Fourier', verb)

    # Reshape for number of sources
    EM = np.squeeze(EM.reshape((-1, nrec, nsrc), order='F'))

    # === 4.  FINISHED ============
    printstartfinish(verb, t0, kcount)

    return EM


def dipole(src, rec, depth, res, freqtime, signal=None, ab=11, aniso=None,
           epermH=None, epermV=None, mpermH=None, mpermV=None, xdirect=True,
           ht='fht', htarg=None, ft='sin', ftarg=None, opt=None, loop=None,
           verb=2):
    """Return the electromagnetic field due to a dipole source.

    Calculate the electromagnetic frequency- or time-domain field due to
    infinitesimal small electric or magnetic dipole source(s), measured by
    infinitesimal small electric or magnetic dipole receiver(s); sources and
    receivers are directed along the principal directions x, y, or z, and all
    sources are at the same depth, as well as all receivers are at the same
    depth.

    Use the functions `bipole` to calculate dipoles with arbitrary angles or
    bipoles of finite length and arbitrary angle.

    The function `dipole` could be replaced by `bipole` (all there is to do is
    translate `ab` into `msrc`, `mrec`, `azimuth`'s and `dip`'s). However,
    `dipole` is kept separately to serve as an example of a simple modelling
    routine that can serve as a template.


    See Also
    --------
    bipole : Electromagnetic field due to an electromagnetic source.
    fem : Electromagnetic frequency-domain response.
    tem : Electromagnetic time-domain response.


    Parameters
    ----------
    src, rec : list of floats or arrays
        Source and receiver coordinates (m): [x, y, z].
        The x- and y-coordinates can be arrays, z is a single value.
        The x- and y-coordinates must have the same dimension.

    depth : list
        Absolute layer interfaces z (m); #depth = #res - 1
        (excluding +/- infinity).

    res : array_like
        Horizontal resistivities rho_h (Ohm.m); #res = #depth + 1.

    freqtime : array_like
        Frequencies f (Hz) if `signal` == None, else times t (s).

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
                          (default: `empymod.filters.key_201_2009()`)
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

    ft : {'sin', 'cos', 'qwe', 'fftlog'}, optional
        Only used if `signal` != None. Flag to choose either the Sine- or
        Cosine-Filter, the Quadrature-With-Extrapolation (QWE), or FFTLog for
        the Fourier transform.  Defaults to 'sin'.

    ftarg : str or filter from empymod.filters or array_like, optional
        Only used if `signal` !=None. Depends on the value for `ft`:
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
              The module `numexpr` uses by default all available cores up to a
              maximum of 8. You can change this behaviour to your desired
              number of threads `nthreads` with
              `numexpr.set_num_threads(nthreads)`.
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

    verb : {0, 1, 2, 3, 4}, optional
        Level of verbosity, default is 2:
            - 0: Print nothing.
            - 1: Print warnings.
            - 2: Print additional runtime and kernel calls
            - 3: Print additional start/stop, condensed parameter information.
            - 4: Print additional full parameter information


    Returns
    -------
    EM : ndarray, (nfreq, nrec, nsrc)
        Frequency- or time-domain EM field (depending on `signal`):
            - If rec is electric, returns E [V/m].
            - If rec is magnetic, returns B [T] (not H [A/m]!).

        In the case of the impulse time-domain response, the unit is further
        divided by seconds [1/s].

        However, source and receiver are normalised. So for instance in the
        electric case the source strength is 1 A and its length is 1 m. So the
        electric field could also be written as [V/(A.m2)].

        The shape of EM is (nfreq, nrec, nsrc). However, single dimensions
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

    # Check times and Fourier Transform arguments, get required frequencies
    # (freq = freqtime if `signal=None`)
    if signal is not None:
        time, freq, ft, ftarg = check_time(freqtime, signal, ft, ftarg, verb)
    else:
        freq = freqtime

    # Check layer parameters
    model = check_model(depth, res, aniso, epermH, epermV, mpermH, mpermV,
                        verb)
    depth, res, aniso, epermH, epermV, mpermH, mpermV, isfullspace = model

    # Check frequency => get etaH, etaV, zetaH, and zetaV
    frequency = check_frequency(freq, res, aniso, epermH, epermV, mpermH,
                                mpermV, verb)
    freq, etaH, etaV, zetaH, zetaV = frequency

    # Check Hankel transform parameters
    ht, htarg = check_hankel(ht, htarg, verb)

    # Check optimization
    optimization = check_opt(opt, loop, ht, htarg, verb)
    use_spline, use_ne_eval, loop_freq, loop_off = optimization

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
           etaV, zetaH, zetaV, xdirect, isfullspace, ht, htarg, use_spline,
           use_ne_eval, msrc, mrec, loop_freq, loop_off)
    EM, kcount, conv = fem(*inp)

    # In case of QWE, print Warning if not converged
    conv_warning(conv, htarg, 'Hankel', verb)

    # Do f->t transform if required
    if signal is not None:
        EM, conv = tem(EM, off, freq, time, signal, ft, ftarg)

        # In case of QWE, print Warning if not converged
        conv_warning(conv, ftarg, 'Fourier', verb)

    # Reshape for number of sources
    EM = np.squeeze(EM.reshape((-1, nrec, nsrc), order='F'))

    # === 4.  FINISHED ============
    printstartfinish(verb, t0, kcount)

    return EM


def gpr(src, rec, depth, res, fc=250, ab=11, gain=None, aniso=None,
        epermH=None, epermV=None, mpermH=None, mpermV=None, xdirect=True,
        ht='fht', htarg=None, opt=None, loop='off', verb=2):
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
    t0 = printstartfinish(verb)

    # === 2.  CHECK INPUT ============

    # Frequency range from centre frequency
    fc *= 10**6
    freq = np.linspace(1, 2048, 2048)*10**6

    # Check layer parameters
    model = check_model(depth, res, aniso, epermH, epermV, mpermH, mpermV,
                        verb)
    depth, res, aniso, epermH, epermV, mpermH, mpermV, isfullspace = model

    # Check frequency => get etaH, etaV, zetaH, and zetaV
    frequency = check_frequency(freq, res, aniso, epermH, epermV, mpermH,
                                mpermV, verb)
    freq, etaH, etaV, zetaH, zetaV = frequency

    # Check Hankel transform parameters
    ht, htarg = check_hankel(ht, htarg, verb)

    # Check optimization
    optimization = check_opt(opt, loop, ht, htarg, verb)
    use_spline, use_ne_eval, loop_freq, loop_off = optimization

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

    # Collect variables for fem
    fdata = (ab_calc, off, angle, zsrc, zrec, lsrc, lrec, depth, freq, etaH,
             etaV, zetaH, zetaV, xdirect, isfullspace, ht, htarg, use_spline,
             use_ne_eval, msrc, mrec, loop_freq, loop_off)

    # === 3. GPR CALCULATION ============

    # 1. Get fem responses
    fEM, kcount, conv = fem(*fdata)
    conv_warning(conv, htarg, 'Hankel', verb)

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
    printstartfinish(verb, t0, kcount)

    return t[2048:], gprEM[2048:, :].real


def wavenumber(src, rec, depth, res, freq, wavenumber, ab=11, aniso=None,
               epermH=None, epermV=None, mpermH=None, mpermV=None,
               xdirect=True, verb=2):
    """Return the electromagnetic wavenumber-domain field.

    Calculate the electromagnetic wavenumber-domain field due to infinitesimal
    small electric or magnetic dipole source(s), measured by infinitesimal
    small electric or magnetic dipole receiver(s); sources and receivers are
    directed along the principal directions x, y, or z, and all sources are at
    the same depth, as well as all receivers are at the same depth.


    See Also
    --------
    dipole : Electromagnetic field due to an electromagnetic source (dipoles).
    bipole : Electromagnetic field due to an electromagnetic source (bipoles).
    fem : Electromagnetic frequency-domain response.
    tem : Electromagnetic time-domain response.


    Parameters
    ----------
    src, rec : list of floats or arrays
        Source and receiver coordinates (m): [x, y, z].
        The x- and y-coordinates can be arrays, z is a single value.
        The x- and y-coordinates must have the same dimension.
        The x- and y-coordinates only matter for the angle-dependent factor.

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

    xdirect : bool, optional
        If True and source and receiver are in the same layer, the direct field
        is calculated analytically in the frequency domain, if False it is
        calculated in the wavenumber domain.
        Defaults to True.

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
    >>> from empymod.model import wavenumber
    >>> src = [0, 0, 100]
    >>> rec = [5000, 0, 200]
    >>> depth = [0, 300, 1000, 1050]
    >>> res = [1e20, .3, 1, 50, 1]
    >>> freq = 1
    >>> wavenrs = np.logspace(-3.7, -3.6, 10)
    >>> PJ0, PJ1 = wavenumber(src, rec, depth, res, freq, wavenrs, verb=0)
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
    modl = check_model(depth, res, aniso, epermH, epermV, mpermH, mpermV, verb)
    depth, res, aniso, epermH, epermV, mpermH, mpermV, _ = modl

    # Check frequency => get etaH, etaV, zetaH, and zetaV
    f = check_frequency(freq, res, aniso, epermH, epermV, mpermH, mpermV, verb)
    _, etaH, etaV, zetaH, zetaV = f  # (output freq not required)

    # Check src-rec configuration
    # => Get flags if src or rec or both are magnetic (msrc, mrec)
    ab_calc, msrc, mrec = check_ab(ab, verb)

    # Check src and rec
    src, nsrc = check_dipole(src, 'src', verb)
    rec, nrec = check_dipole(rec, 'rec', verb)

    # Get angle-dependent factor
    _, angle = get_off_ang(src, rec, nsrc, nrec, verb)
    factAng = kernel.angle_factor(angle, ab, msrc, mrec)

    # Get layer number in which src and rec reside (lsrc/lrec)
    lsrc, zsrc = get_layer_nr(src, depth)
    lrec, zrec = get_layer_nr(rec, depth)

    # === 3. EM-FIELD CALCULATION ============

    # Calculate wavenumber response
    PJ0, PJ1, PJ0b = kernel.wavenumber(zsrc, zrec, lsrc, lrec, depth, etaH,
                                       etaV, zetaH, zetaV,
                                       np.atleast_2d(wavenumber), ab_calc,
                                       xdirect, msrc, mrec, False)

    # Collect output
    PJ1 = np.squeeze(factAng[:, np.newaxis]*PJ1*wavenumber)
    PJ0 = np.squeeze(PJ0 + factAng[:, np.newaxis]*PJ0b)

    # === 4.  FINISHED ============
    printstartfinish(verb, t0, 1)

    return PJ0, PJ1


# Shortcuts (legacy routines)

def frequency(src, rec, depth, res, freq, ab=11, aniso=None, epermH=None,
              epermV=None, mpermH=None, mpermV=None, xdirect=True, ht='fht',
              htarg=None, opt=None, loop=None, verb=2):
    """Return the frequency-domain EM field due to a dipole source.

    This is a shortcut for frequency-domain modelling using `dipole` (mainly
    for legacy reasons).

    See `dipole` for info and a description of input and output parameters.
    Only difference is that `frequency` here corresponds to `freqtime` in
    `dipole`.


    See Also
    --------
    dipole : EM field due to an EM source (dipole-dipole).
    bipole : EM field due to an EM source (bipole-bipole).


    Examples
    --------
    >>> import numpy as np
    >>> from empymod import frequency
    >>> src = [0, 0, 100]
    >>> rec = [np.arange(1, 11)*500, np.zeros(10), 200]
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

    return dipole(src, rec, depth, res, freq, None, ab, aniso, epermH, epermV,
                  mpermH, mpermV, xdirect, ht, htarg, opt=opt, loop=loop,
                  verb=verb)


def time(src, rec, depth, res, time, ab=11, signal=0, aniso=None, epermH=None,
         epermV=None, mpermH=None, mpermV=None, xdirect=True, ht='fht',
         htarg=None, ft='sin', ftarg=None, opt=None, loop='off', verb=2):
    """Return the time-domain EM field due to a dipole source.

    This is a shortcut for time-domain modelling using `dipole` (mainly for
    legacy reasons).

    See `dipole` for info and a description of input and output parameters.
    Only difference is that `time` here corresponds to `freqtime` in `dipole`.


    See Also
    --------
    dipole : EM field due to an EM source (dipole-dipole).
    bipole : EM field due to an EM source (bipole-bipole).


    Examples
    --------
    >>> import numpy as np
    >>> from empymod import time
    >>> src = [0, 0, 100]
    >>> rec = [np.arange(1, 11)*500, np.zeros(10), 200]
    >>> depth = [0, 300, 1000, 1050]
    >>> res = [1e20, .3, 1, 50, 1]
    >>> EMfield = time(src, rec, depth, res, time=1, verb=0)
    >>> print(EMfield)
    [  4.23754930e-11   3.13805193e-11   1.98884433e-11   1.14387827e-11
       6.34605628e-12   3.54905259e-12   2.03906739e-12   1.20569287e-12
       7.31746271e-13   4.55825907e-13]

    """

    return dipole(src, rec, depth, res, time, signal, ab, aniso, epermH,
                  epermV, mpermH, mpermV, xdirect, ht, htarg, ft, ftarg, opt,
                  loop, verb)


# Core modelling routines

def fem(ab, off, angle, zsrc, zrec, lsrc, lrec, depth, freq, etaH, etaV, zetaH,
        zetaV, xdirect, isfullspace, ht, htarg, use_spline, use_ne_eval, msrc,
        mrec, loop_freq, loop_off, conv=True):
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

    # Initialize kernel count
    # (how many times the wavenumber-domain kernel was calld)
    kcount = 0

    # If <ab> = 36 (or 63), fEM-field is zero
    if ab in [36, ]:
        return fEM, kcount, conv

    # Get full-space-solution if model is a full-space or
    # if src and rec are in the same layer and xdirect=True.
    if isfullspace or (lsrc == lrec and xdirect):
        fEM += kernel.fullspace(off, angle, zsrc, zrec, etaH[:, lrec],
                                etaV[:, lrec], zetaH[:, lrec], zetaV[:, lrec],
                                ab, msrc, mrec)

    # If not full-space calculate fEM-field
    if not isfullspace:
        calc = getattr(transform, ht)
        if loop_freq:

            for i in range(freq.size):
                out = calc(zsrc, zrec, lsrc, lrec, off, angle, depth, ab,
                           etaH[None, i, :], etaV[None, i, :],
                           zetaH[None, i, :], zetaV[None, i, :], xdirect,
                           htarg, use_spline, use_ne_eval, msrc, mrec)
                fEM[None, i, :] += out[0]
                kcount += out[1]
                conv *= out[2]

        elif loop_off:
            for i in range(off.size):
                out = calc(zsrc, zrec, lsrc, lrec, off[None, i],
                           angle[None, i], depth, ab, etaH, etaV, zetaH, zetaV,
                           xdirect, htarg, use_spline, use_ne_eval, msrc, mrec)
                fEM[:, None, i] += out[0]
                kcount += out[1]
                conv *= out[2]
        else:
            out = calc(zsrc, zrec, lsrc, lrec, off, angle, depth, ab, etaH,
                       etaV, zetaH, zetaV, xdirect, htarg, use_spline,
                       use_ne_eval, msrc, mrec)
            fEM += out[0]
            kcount += out[1]
            conv *= out[2]

    return fEM, kcount, conv


def tem(fEM, off, freq, time, signal, ft, ftarg, conv=True):
    """Return the time-domain response of the frequency-domain response fEM.

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
