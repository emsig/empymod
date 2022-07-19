"""
EM-modelling routines. The implemented routines might not be the fastest
solution to your specific problem. Use these routines as template to create
your own, problem-specific modelling routine!

Principal routines:

- :func:`bipole`
- :func:`dipole`
- :func:`loop`

The main routine is :func:`bipole`, which can model bipole source(s) and bipole
receiver(s) of arbitrary direction, for electric or magnetic sources and
receivers, both in frequency and in time. A subset of :func:`bipole` is
:func:`dipole`, which models infinitesimal small dipoles along the principal
axes x, y, and z. The third routine, :func:`loop`, can be used if the source or
the receivers are loops instead of dipoles.

Further routines are:

- :func:`analytical`: Calculate analytical fullspace and halfspace solutions.
- :func:`dipole_k`: Calculate the electromagnetic wavenumber-domain solution.
- :func:`gpr`: Calculate the Ground-Penetrating Radar (GPR) response.

The :func:`dipole_k` routine can be used if you are interested in the
wavenumber-domain result, without Hankel nor Fourier transform. It calls
straight the :mod:`empymod.kernel`. The :func:`gpr`-routine convolves the
frequency-domain result with a wavelet, and applies a gain to the time-domain
result. This function is still experimental.

The modelling routines make use of the following two core routines:

- :func:`fem`: Calculate wavenumber-domain electromagnetic field and carry out
  the Hankel transform to the frequency domain.
- :func:`tem`: Carry out the Fourier transform to time domain after
  :func:`fem`.

"""
# Copyright 2016-2022 The emsig community.
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


import numpy as np

from empymod import kernel, transform
from empymod.utils import (
        check_time, check_time_only, check_model, check_frequency,
        check_hankel, check_loop, check_dipole, check_bipole, check_ab,
        check_solution, get_abs, get_geo_fact, get_azm_dip, get_off_ang,
        get_layer_nr, get_kwargs, printstartfinish, conv_warning, EMArray)

__all__ = ['bipole', 'dipole', 'loop', 'analytical', 'gpr', 'dipole_k', 'fem',
           'tem']


def bipole(src, rec, depth, res, freqtime, signal=None, aniso=None,
           epermH=None, epermV=None, mpermH=None, mpermV=None, msrc=False,
           srcpts=1, mrec=False, recpts=1, strength=0, **kwargs):
    r"""Return EM fields due to arbitrary rotated, finite length EM dipoles.

    Calculate the electromagnetic frequency- or time-domain field due to
    arbitrary rotated, finite electric or magnetic bipole sources, measured by
    arbitrary rotated, finite electric or magnetic bipole receivers. By
    default, the electromagnetic response is normalized to source and receiver
    of 1 m length, and source strength of 1 A.


    See Also
    --------
    :func:`dipole` : EM fields due to infinitesimal small EM dipoles.
    :func:`loop` : EM fields due to a magnetic source loop.


    Parameters
    ----------
    src, rec : list of floats or arrays
        Source and receiver coordinates (m):

        - [x0, x1, y0, y1, z0, z1] (bipole of finite length)
        - [x, y, z, azimuth, dip]  (dipole, infinitesimal small)

        Dimensions:

        - The coordinates x, y, and z (dipole) or x0, x1, y0, y1, z0, and z1
          (bipole) can be single values or arrays.
        - The variables x and y (dipole) or x0, x1, y0, and y1 (bipole) must
          have the same dimensions.
        - The variables z, azimuth, and dip (dipole) or z0 and z1 (bipole) must
          either be single values or having the same dimension as the other
          coordinates.

        Angles (coordinate system is either left-handed with positive z down or
        right-handed with positive z up; East-North-Depth):

        - azimuth (°): horizontal deviation from x-axis, anti-clockwise.
        - +/-dip (°): vertical deviation from xy-plane down/up-wards.

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
        Frequencies f (Hz) if `signal==None`, else times t (s); (f, t > 0).

    signal : {None, 0, 1, -1}, default: None
        Source signal:

        - None: Frequency-domain response
        - -1 : Switch-off time-domain response
        - 0 : Impulse time-domain response
        - +1 : Switch-on time-domain response

    aniso : array_like, default: ones
        Anisotropies lambda = sqrt(rho_v/rho_h) (-); #aniso = #res.

    epermH, epermV : array_like, default: ones
        Relative horizontal/vertical electric permittivities
        epsilon_h/epsilon_v (-); #epermH = #epermV = #res. If epermH is
        provided but not epermV, isotropic behaviour is assumed.

    mpermH, mpermV : array_like, default: ones
        Relative horizontal/vertical magnetic permeabilities mu_h/mu_v (-);
        #mpermH = #mpermV = #res. If mpermH is provided but not mpermV,
        isotropic behaviour is assumed.

    msrc, mrec : bool, default: False
        If True, source/receiver (msrc/mrec) is magnetic, else electric.

    srcpts, recpts : int, default: 1
        Number of integration points for bipole source/receiver:

        - srcpts/recpts < 3  : bipole, but calculated as dipole at centre
        - srcpts/recpts >= 3 : bipole

    strength : float, default: 0.0
        Source strength (A):

        - If 0, output is normalized to source and receiver of 1 m length, and
          source strength of 1 A.
        - If != 0, output is returned for given source and receiver length, and
          source strength.

    verb : {0, 1, 2, 3, 4}, default: 2
        Level of verbosity:

        - 0: Print nothing.
        - 1: Print warnings.
        - 2: Print additional runtime and kernel calls
        - 3: Print additional start/stop, condensed parameter information.
        - 4: Print additional full parameter information

    ht : {'dlf', 'qwe', 'quad'}, default: 'dlf'
        Flag to choose either the *Digital Linear Filter* (DLF) method, the
        *Quadrature-With-Extrapolation* (QWE), or a simple *Quadrature* (QUAD)
        for the Hankel transform.

    htarg : dict, optional
        Possible parameters depends on the value for `ht`:

        - If `ht='dlf'`:

          - `dlf`: string of filter name in :mod:`empymod.filters` or the
            filter method itself. (default:
            :func:`empymod.filters.key_201_2009`)
          - `pts_per_dec`: points per decade; (default: 0):

            - If 0: Standard DLF.
            - If < 0: Lagged Convolution DLF.
            - If > 0: Splined DLF

        - If `ht='qwe'`:

          - `rtol`: relative tolerance (default: 1e-12)
          - `atol`: absolute tolerance (default: 1e-30)
          - `nquad`: order of Gaussian quadrature (default: 51)
          - `maxint`: maximum number of partial integral intervals
            (default: 40)
          - `pts_per_dec`: points per decade; (default: 0)

            - If 0, no interpolation is used.
            - If > 0, interpolation is used.

          - `diff_quad`: criteria when to swap to QUAD (only relevant if
            pts_per_dec=-1) (default: 100)
          - `a`: lower limit for QUAD (default: first interval from QWE)
          - `b`: upper limit for QUAD (default: last interval from QWE)
          - `limit`: limit for quad (default: maxint)

        - If `ht='quad'`:

          - `rtol`: relative tolerance (default: 1e-12)
          - `atol`: absolute tolerance (default: 1e-20)
          - `limit`: An upper bound on the number of subintervals used in the
            adaptive algorithm (default: 500)
          - `a`: Minimum wavenumber (default 1e-6)
          - `b`: Maximum wavenumber (default 0.1)
          - `pts_per_dec`: points per decade (default: 40)

    ft : {'dlf', 'sin', 'cos', 'qwe', 'fftlog', 'fft'}, default: 'dlf'
        Only used if signal!=None. Flag to choose either the Digital Linear
        Filter method (Sine- or Cosine-Filter), the
        Quadrature-With-Extrapolation (QWE), the FFTLog, or the FFT for the
        Fourier transform. If 'dlf' it is 'sin' if signal>=0, else 'cos'.

    ftarg : dict, optional
        Only used if signal!=None. Possible parameters depends on the value for
        `ft`:

        - If `ft='dlf'`, 'sin', or 'cos':

          - `dlf`: string of filter name in :mod:`empymod.filters` or the
            filter method itself. (Default:
            :func:`empymod.filters.key_201_CosSin_2012`)
          - `pts_per_dec`: points per decade; (default: -1)

            - If 0: Standard DLF.
            - If < 0: Lagged Convolution DLF.
            - If > 0: Splined DLF


        - If `ft='qwe'`:

          - `rtol`: relative tolerance (default: 1e-8)
          - `atol`: absolute tolerance (default: 1e-20)
          - `nquad`: order of Gaussian quadrature (default: 21)
          - `maxint`: maximum number of partial integral intervals
            (default: 200)
          - `pts_per_dec`: points per decade (default: 20)
          - `diff_quad`: criteria when to swap to QUAD (default: 100)
          - `a`: lower limit for QUAD (default: first interval from QWE)
          - `b`: upper limit for QUAD (default: last interval from QWE)
          - `limit`: limit for quad (default: maxint)

        - If `ft='fftlog'`:

          - `pts_per_dec`: sampels per decade (default: 10)
          - `add_dec`: additional decades [left, right] (default: [-2, 1])
          - `q`: exponent of power law bias (default: 0); -1 <= q <= 1

        - If `ft='fft'`:

          - `dfreq`: Linear step-size of frequencies (default: 0.002)
          - `nfreq`: Number of frequencies (default: 2048)
          - `ntot`: Total number for FFT; difference between nfreq and ntot
            is padded with zeroes. This number is ideally a power of 2, e.g.
            2048 or 4096 (default: nfreq).
          - `pts_per_dec`: points per decade (default: None)

          Padding can sometimes improve the result, not always. The default
          samples from 0.002 Hz - 4.096 Hz. If pts_per_dec is set to an
          integer, calculated frequencies are logarithmically spaced with the
          given number per decade, and then interpolated to yield the required
          frequencies for the FFT.

    xdirect : bool or None, default: False
        Direct field calculation (only if src and rec are in the same layer):

        - If True, direct field is calculated analytically in the frequency
          domain.
        - If False, direct field is calculated in the wavenumber domain.
        - If None, direct field is excluded from the calculation, and only
          reflected fields are returned (secondary field).

    loop : {None, 'freq', 'off'}, default: None
        Define if to calculate everything vectorized or if to loop over
        frequencies ('freq') or over offsets ('off'). It always loops over
        frequencies if `ht='qwe'` or if `pts_per_dec=-1`. Calculating
        everything vectorized is fast for few offsets OR for few frequencies.
        However, if you calculate many frequencies for many offsets, it might
        be faster to loop over frequencies. Only comparing the different
        versions will yield the answer for your specific problem at hand!

    squeeze : bool, default: True
        If True, the output is squeezed. If False, the output will always be of
        ``ndim=3``, (nfreqtime, nrec, nsrc).


    Returns
    -------
    EM : EMArray, (nfreqtime, nrec, nsrc)
        Frequency- or time-domain EM field (depending on `signal`):

        - If rec is electric, returns E [V/m].
        - If rec is magnetic, returns H [A/m].

        EMArray is a subclassed ndarray with `.pha` and `.amp` attributes
        (only relevant for frequency-domain data).

        The shape of EM is (nfreqtime, nrec, nsrc). However, single dimensions
        are removed.


    Examples
    --------

    .. ipython::

       In [1]: import empymod
          ...: import numpy as np
          ...: # x-directed bipole source: x0, x1, y0, y1, z0, z1
          ...: src = [-50, 50, 0, 0, 100, 100]
          ...: # x-directed dipole receiver-array: x, y, z, azimuth, dip
          ...: rec = [np.arange(1, 11)*500, np.zeros(10), 200, 0, 0]
          ...: # layer boundaries
          ...: depth = [0, 300, 1000, 1050]
          ...: # layer resistivities
          ...: res = [1e20, .3, 1, 50, 1]
          ...: # Frequency
          ...: freq = 1
          ...: # Calculate electric field due to an electric source at 1 Hz.
          ...: # [msrc = mrec = False (default)]
          ...: EMfield = empymod.bipole(src, rec, depth, res, freq, verb=3)
       Out[1]:
          ...: :: empymod START  ::  v2.0.0
          ...:
          ...:    depth       [m] :  0 300 1000 1050
          ...:    res     [Ohm.m] :  1E+20 0.3 1 50 1
          ...:    aniso       [-] :  1 1 1 1 1
          ...:    epermH      [-] :  1 1 1 1 1
          ...:    epermV      [-] :  1 1 1 1 1
          ...:    mpermH      [-] :  1 1 1 1 1
          ...:    mpermV      [-] :  1 1 1 1 1
          ...:    direct field    :  Comp. in wavenumber domain
          ...:    frequency  [Hz] :  1
          ...:    Hankel          :  DLF (Fast Hankel Transform)
          ...:      > Filter      :  Key 201 (2009)
          ...:      > DLF type    :  Standard
          ...:    Loop over       :  None (all vectorized)
          ...:    Source(s)       :  1 bipole(s)
          ...:      > intpts      :  1 (as dipole)
          ...:      > length  [m] :  100
          ...:      > strength[A] :  0
          ...:      > x_c     [m] :  0
          ...:      > y_c     [m] :  0
          ...:      > z_c     [m] :  100
          ...:      > azimuth [°] :  0
          ...:      > dip     [°] :  0
          ...:    Receiver(s)     :  10 dipole(s)
          ...:      > x       [m] :  500 - 5000 : 10  [min-max; #]
          ...:      > y       [m] :  0 - 0 : 10  [min-max; #]
          ...:      > z       [m] :  200
          ...:      > azimuth [°] :  0
          ...:      > dip     [°] :  0
          ...:    Required ab's   :  11
          ...:
          ...: :: empymod END; runtime = 0:00:00.005536 :: 1 kernel call(s)

       In [2]: EMfield[0]
       Out[2]: (1.6880934577857306e-10-3.083031298956568e-10j)

    """
    # Get kwargs with defaults.
    out = get_kwargs(
        ['verb', 'ht', 'htarg', 'ft', 'ftarg', 'xdirect', 'loop', 'squeeze'],
        [2, 'dlf', {}, 'dlf', {}, False, None, True], kwargs,
    )
    verb, ht, htarg, ft, ftarg, xdirect, loop, squeeze = out

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

    # Check loop
    loop_freq, loop_off = check_loop(loop, ht, htarg, verb)

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

    # Define some indices
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
                            htarg, msrc, mrec, loop_freq, loop_off, conv)

                    # Pre-allocate temporary EM array for ab-loop
                    abEM = np.zeros((freq.size, isrz), dtype=etaH.dtype)

                    for iab in ab_calc:  # Loop over required ab's

                        # Carry-out the frequency-domain calculation
                        out = fem(iab, *finp)

                        # Get geometrical scaling factor,
                        # broadcast to (irec, isrc)
                        tfact = np.ones((irec, isrc))*get_geo_fact(
                            iab, srcazm, srcdip, recazm, recdip, msrc, mrec
                        )

                        # Add field to EM with geometrical factor
                        abEM += out[0]*tfact.ravel('F')

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
    EM = EM.reshape((-1, nrec, nsrc), order='F')
    if squeeze:
        EM = np.squeeze(EM)

    # === 4.  FINISHED ============
    printstartfinish(verb, t0, kcount)

    return EMArray(EM)


def dipole(src, rec, depth, res, freqtime, signal=None, ab=11, aniso=None,
           epermH=None, epermV=None, mpermH=None, mpermV=None, **kwargs):
    r"""Return EM fields due to infinitesimal small EM dipoles.

    Calculate the electromagnetic frequency- or time-domain field due to
    infinitesimal small electric or magnetic dipole source(s), measured by
    infinitesimal small electric or magnetic dipole receiver(s); sources and
    receivers are directed along the principal directions x, y, or z, and all
    sources are at the same depth, as well as all receivers are at the same
    depth.

    Use the functions :func:`bipole` to calculate dipoles with arbitrary angles
    or bipoles of finite length and arbitrary angle.

    The function :func:`dipole` could be replaced by :func:`bipole` (all there
    is to do is translate `ab` into `msrc`, `mrec`, `azimuth`'s and `dip`'s).
    However, :func:`dipole` is kept separately to serve as an example of a
    simple modelling routine that can serve as a template.


    See Also
    --------
    :func:`bipole` : EM fields due to arbitrary rotated, finite length EM
                     dipoles.
    :func:`loop` : EM fields due to a magnetic source loop.


    Parameters
    ----------
    src, rec : list of floats or arrays
        Source and receiver coordinates [x, y, z] (m):

        - The x- and y-coordinates can be arrays, z is a single value.
        - The x- and y-coordinates must have the same dimension.

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
        Frequencies f (Hz) if `signal==None`, else times t (s); (f, t > 0).

    signal : {None, 0, 1, -1}, default: None
        Source signal:

        - None: Frequency-domain response
        - -1 : Switch-off time-domain response
        - 0 : Impulse time-domain response
        - +1 : Switch-on time-domain response

    ab : int, default: 11
        Source-receiver configuration.

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

    aniso : array_like, default: ones
        Anisotropies lambda = sqrt(rho_v/rho_h) (-); #aniso = #res.

    epermH, epermV : array_like, default: ones
        Relative horizontal/vertical electric permittivities
        epsilon_h/epsilon_v (-); #epermH = #epermV = #res. If epermH is
        provided but not epermV, isotropic behaviour is assumed.

    mpermH, mpermV : array_like, default: ones
        Relative horizontal/vertical magnetic permeabilities mu_h/mu_v (-);
        #mpermH = #mpermV = #res. If mpermH is provided but not mpermV,
        isotropic behaviour is assumed.

    verb : {0, 1, 2, 3, 4}, default: 2
        Level of verbosity:

        - 0: Print nothing.
        - 1: Print warnings.
        - 2: Print additional runtime and kernel calls
        - 3: Print additional start/stop, condensed parameter information.
        - 4: Print additional full parameter information

    ht, htarg, ft, ftarg, xdirect, loop : settings, optinal
        See docstring of :func:`bipole` for a description.

    squeeze : bool, default: True
        If True, the output is squeezed. If False, the output will always be of
        ``ndim=3``, (nfreqtime, nrec, nsrc).


    Returns
    -------
    EM : EMArray, (nfreqtime, nrec, nsrc)
        Frequency- or time-domain EM field (depending on `signal`):

        - If rec is electric, returns E [V/m].
        - If rec is magnetic, returns H [A/m].

        EMArray is a subclassed ndarray with `.pha` and `.amp` attributes
        (only relevant for frequency-domain data).

        The shape of EM is (nfreqtime, nrec, nsrc). However, single dimensions
        are removed.


    Examples
    --------

    .. ipython::

       In [1]: import empymod
          ...: import numpy as np
          ...: src = [0, 0, 100]
          ...: rec = [np.arange(1, 11)*500, np.zeros(10), 200]
          ...: depth = [0, 300, 1000, 1050]
          ...: res = [1e20, .3, 1, 50, 1]
          ...: EMfield = empymod.dipole(
          ...:         src, rec, depth, res, freqtime=1, verb=1)
          ...: EMfield[0]
       Out[1]: (1.6880934577857306e-10-3.083031298956568e-10j)

    """
    # Get kwargs with defaults.
    out = get_kwargs(
        ['verb', 'ht', 'htarg', 'ft', 'ftarg', 'xdirect', 'loop', 'squeeze'],
        [2, 'dlf', {}, 'dlf', {}, False, None, True], kwargs,
    )
    verb, ht, htarg, ft, ftarg, xdirect, loop, squeeze = out

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

    # Check loop
    loop_freq, loop_off = check_loop(loop, ht, htarg, verb)

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
           etaV, zetaH, zetaV, xdirect, isfullspace, ht, htarg, msrc, mrec,
           loop_freq, loop_off)
    EM, kcount, conv = fem(*inp)

    # In case of QWE/QUAD, print Warning if not converged
    conv_warning(conv, htarg, 'Hankel', verb)

    # Do f->t transform if required
    if signal is not None:
        EM, conv = tem(EM, off, freq, time, signal, ft, ftarg)

        # In case of QWE/QUAD, print Warning if not converged
        conv_warning(conv, ftarg, 'Fourier', verb)

    # Reshape for number of sources
    EM = EM.reshape((-1, nrec, nsrc), order='F')
    if squeeze:
        EM = np.squeeze(EM)

    # === 4.  FINISHED ============
    printstartfinish(verb, t0, kcount)

    return EMArray(EM)


def loop(src, rec, depth, res, freqtime, signal=None, aniso=None, epermH=None,
         epermV=None, mpermH=None, mpermV=None, mrec=True, recpts=1,
         strength=0, **kwargs):
    r"""Return EM fields due to a magnetic source loop.

    Calculate the electromagnetic frequency- or time-domain field due to
    an arbitrary rotated, magnetic source consisting of an electric loop,
    measured by arbitrary rotated, finite electric or magnetic bipole
    receivers or arbitrary rotated magnetic receivers consisting of electric
    loops. By default, the electromagnetic response is normalized to source
    loop area of 1 m2 and receiver length or area of 1 m or 1 m2, respectively,
    and source strength of 1 A.

    A magnetic dipole, as used in :func:`dipole` and :func:`bipole`, has a
    moment of :math:`I^m ds`. However, if the magnetic dipole is generated by
    an electric-wire loop, this changes to :math:`I^m = i\omega\mu A I^e`,
    where A is the area of the loop. The same factor :math:`i\omega\mu A`,
    applies to the receiver, if it consists of an electric-wire loop.

    The current implementation only handles loop sources and receivers in
    layers where :math:`\mu_r^h=\mu_r^v`; the horizontal magnetic permeability
    is used, and a warning is thrown if the vertical differs from the
    horizontal one.

    Note that the kernel internally still calculates dipole sources and
    receivers, the moment is a factor that is multiplied in the frequency
    domain. The logs will therefore still inform about bipoles and dipoles.


    See Also
    --------
    :func:`dipole` : EM fields due to infinitesimal small EM dipoles.
    :func:`bipole` : EM fields due to arbitrary rotated, finite length EM
                    dipoles.


    Parameters
    ----------
    src, rec : list of floats or arrays
        Source and receiver coordinates (m):

        - [x0, x1, y0, y1, z0, z1] (bipole of finite length)
        - [x, y, z, azimuth, dip]  (dipole, infinitesimal small)

        Dimensions:

        - The coordinates x, y, and z (dipole) or x0, x1, y0, y1, z0, and z1
          (bipole) can be single values or arrays.
        - The variables x and y (dipole) or x0, x1, y0, and y1 (bipole) must
          have the same dimensions.
        - The variables z, azimuth, and dip (dipole) or z0 and z1 (bipole) must
          either be single values or having the same dimension as the other
          coordinates.

        Angles (coordinate system is either left-handed with positive z down or
        right-handed with positive z up; East-North-Depth):

        - azimuth (°): horizontal deviation from x-axis, anti-clockwise.
        - +/-dip (°): vertical deviation from xy-plane down/up-wards.

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
        Frequencies f (Hz) if `signal==None`, else times t (s); (f, t > 0).

    signal : {None, 0, 1, -1}, default: None
        Source signal:

        - None: Frequency-domain response
        - -1 : Switch-off time-domain response
        - 0 : Impulse time-domain response
        - +1 : Switch-on time-domain response

    aniso : array_like, default: ones
        Anisotropies lambda = sqrt(rho_v/rho_h) (-); #aniso = #res.

    epermH, epermV : array_like, default: ones
        Relative horizontal/vertical electric permittivities
        epsilon_h/epsilon_v (-); #epermH = #epermV = #res. If epermH is
        provided but not epermV, isotropic behaviour is assumed.

    mpermH, mpermV : array_like, default: ones
        Relative horizontal/vertical magnetic permeabilities mu_h/mu_v (-);
        #mpermH = #mpermV = #res. If mpermH is provided but not mpermV,
        isotropic behaviour is assumed.

        Note that the relative horizontal and vertical magnetic permeabilities
        in layers with loop sources or receivers will be set to 1.

    mrec : bool or string, default: True
        Receiver options:

        - True: Magnetic dipole receiver;
        - False: Electric dipole receiver;
        - 'loop': Magnetic receiver consisting of an electric-wire loop.

    recpts : int, default: 1
        Number of integration points for bipole receiver:

        - recpts < 3  : bipole, but calculated as dipole at centre
        - recpts >= 3 : bipole

        Note that if `mrec='loop'`, `recpts` will be set to 1.

    strength : float, default: 0.0
        Source strength (A):

        - If 0, output is normalized to source of 1 m2 area and receiver of 1 m
          length or 1 m2 area, and source strength of 1 A.
        - If != 0, output is returned for given source strength and receiver
          length (if `mrec!='loop'`).

        The strength is simply a multiplication factor. It can also be used to
        provide the source and receiver loop area, or also to multiply by
        :math:\mu_0`, if you want the B-field instead of the H-field.

    verb : {0, 1, 2, 3, 4}, default: 2
        Level of verbosity:

        - 0: Print nothing.
        - 1: Print warnings.
        - 2: Print additional runtime and kernel calls
        - 3: Print additional start/stop, condensed parameter information.
        - 4: Print additional full parameter information

    ht, htarg, ft, ftarg, xdirect, loop : settings, optinal
        See docstring of :func:`bipole` for a description.

    squeeze : bool, default: True
        If True, the output is squeezed. If False, the output will always be of
        ``ndim=3``, (nfreqtime, nrec, nsrc).


    Returns
    -------
    EM : EMArray, (nfreqtime, nrec, nsrc)
        Frequency- or time-domain EM field (depending on `signal`):

        - If rec is electric, returns E [V/m].
        - If rec is magnetic, returns H [A/m].

        EMArray is a subclassed ndarray with `.pha` and `.amp` attributes
        (only relevant for frequency-domain data).


    Examples
    --------

    .. ipython::

       In [1]: import empymod
          ...: import numpy as np
          ...: # z-directed loop source: x, y, z, azimuth, dip
          ...: src = [0, 0, 0, 0, 90]
          ...: # z-directed magn. dipole receiver-array: x, y, z, azimuth, dip
          ...: rec = [np.arange(1, 11)*500, np.zeros(10), 200, 0, 90]
          ...: # layer boundaries
          ...: depth = [0, 300, 500]
          ...: # layer resistivities
          ...: res = [2e14, 10, 500, 10]
          ...: # Frequency
          ...: freq = 1
          ...: # Calculate magnetic field due to a loop source at 1 Hz.
          ...: # [mrec = True (default)]
          ...: EMfield = empymod.loop(src, rec, depth, res, freq, verb=3)
       Out[1]:
          ...: :: empymod START  ::  w2.0.0
          ...:    depth       [m] :  0 300 500
          ...:    res     [Ohm.m] :  2E+14 10 500 10
          ...:    aniso       [-] :  1 1 1 1
          ...:    epermH      [-] :  1 1 1 1
          ...:    epermV      [-] :  1 1 1 1
          ...:    mpermH      [-] :  1 1 1 1
          ...:    mpermV      [-] :  1 1 1 1
          ...:    direct field    :  Comp. in wavenumber domain
          ...:    frequency  [Hz] :  1
          ...:    Hankel          :  DLF (Fast Hankel Transform)
          ...:      > Filter      :  Key 201 (2009)
          ...:      > DLF type    :  Standard
          ...:    Loop over       :  None (all vectorized)
          ...:    Source(s)       :  1 dipole(s)
          ...:      > x       [m] :  0
          ...:      > y       [m] :  0
          ...:      > z       [m] :  0
          ...:      > azimuth [°] :  0
          ...:      > dip     [°] :  90
          ...:    Receiver(s)     :  10 dipole(s)
          ...:      > x       [m] :  500 - 5000 : 10  [min-max; #]
          ...:      > y       [m] :  0 - 0 : 10  [min-max; #]
          ...:      > z       [m] :  200
          ...:      > azimuth [°] :  0
          ...:      > dip     [°] :  90
          ...:    Required ab's   :  33
          ...:
          ...: :: empymod END; runtime = 0:00:00.005025 :: 1 kernel call(s)

       In [2]: EMfield[0]
       Out[2]: (-3.054498478653836e-10-2.0037418529368025e-11j)

    """
    # Get kwargs with defaults.
    out = get_kwargs(
        ['verb', 'ht', 'htarg', 'ft', 'ftarg', 'xdirect', 'loop', 'squeeze'],
        [2, 'dlf', {}, 'dlf', {}, False, None, True], kwargs,
    )
    verb, ht, htarg, ft, ftarg, xdirect, loop, squeeze = out

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

    # Check loop
    loop_freq, loop_off = check_loop(loop, ht, htarg, verb)

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

    # Define some indices
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
                print("* WARNING :: `mpermH != mpermV` at source level, "
                      "only `mpermH` considered for loop factor.")

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
                    print("* WARNING :: `mpermH != mpermV` at receiver level, "
                          "only `mpermH` considered for loop factor.")

                # Gather variables
                finp = (off, angle, zsrc, zrec, lsrc, lrec, depth, freq,
                        etaH, etaV, zetaH, zetaV, xdirect, isfullspace, ht,
                        htarg, True, mrec, loop_freq, loop_off, conv)

                # Pre-allocate temporary EM array for ab-loop
                abEM = np.zeros((freq.size, isrz), dtype=etaH.dtype)

                for iab in ab_calc:  # Loop over required ab's

                    # Carry-out the frequency-domain calculation
                    out = fem(iab, *finp)

                    # Get geometrical scaling factor, broadcast to (irec, isrc)
                    tfact = np.ones((irec, isrc))*get_geo_fact(
                        iab, srcazm, srcdip, recazm, recdip, True, mrec
                    )

                    # Add field to EM with geometrical factor
                    abEM += out[0]*tfact.ravel('F')

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
    EM = EM.reshape((-1, nrec, nsrc), order='F')
    if squeeze:
        EM = np.squeeze(EM)

    # === 4.  FINISHED ============
    printstartfinish(verb, t0, kcount)

    return EMArray(EM)


def analytical(src, rec, res, freqtime, solution='fs', signal=None, ab=11,
               aniso=None, epermH=None, epermV=None, mpermH=None, mpermV=None,
               **kwargs):
    r"""Return analytical full- or half-space solution.

    Calculate the electromagnetic frequency- or time-domain field due to
    infinitesimal small electric or magnetic dipole source(s), measured by
    infinitesimal small electric or magnetic dipole receiver(s); sources and
    receivers are directed along the principal directions x, y, or z, and all
    sources are at the same depth, as well as all receivers are at the same
    depth.

    In the case of a halfspace the air-interface is located at z = 0 m.

    You can call the functions :func:`empymod.kernel.fullspace` and
    :func:`empymod.kernel.halfspace` in :mod:`empymod.kernel` directly. This
    interface is just to provide a consistent interface with the same input
    parameters as for instance for :func:`dipole`.

    This function yields the same result if `solution='fs'` as :func:`dipole`,
    if the model is a fullspace.

    Included are:

    - Full fullspace solution (`solution='fs'`) for ee-, me-, em-, mm-fields,
      only frequency domain, [HuTS15]_.
    - Diffusive fullspace solution (`solution='dfs'`) for ee-fields, [SlHM10]_.
    - Diffusive halfspace solution (`solution='dhs'`) for ee-fields, [SlHM10]_.
    - Diffusive direct- and reflected field and airwave (`solution='dsplit'`)
      for ee-fields, [SlHM10]_.
    - Diffusive direct- and reflected field and airwave (`solution='dtetm'`)
      for ee-fields, split into TE and TM mode [SlHM10]_.

    Parameters
    ----------
    src, rec : list of floats or arrays
        Source and receiver coordinates [x, y, z] (m):

        - The x- and y-coordinates can be arrays, z is a single value.
        - The x- and y-coordinates must have the same dimension.

    res : float
        Horizontal resistivity rho_h (Ohm.m).

        Alternatively, res can be a dictionary. See the main manual of empymod
        too see how to exploit this hook to re-calculate etaH, etaV, zetaH, and
        zetaV, which can be used to, for instance, use the Cole-Cole model for
        IP.

    freqtime : array_like
        Frequencies f (Hz) if `signal==None`, else times t (s); (f, t > 0).

    solution : str, default: 'fs'
      Defines which solution is returned:

      - 'fs' : Full fullspace solution (ee-, me-, em-, mm-fields); f-domain.
      - 'dfs' : Diffusive fullspace solution (ee-fields only).
      - 'dhs' : Diffusive halfspace solution (ee-fields only).
      - 'dsplit' : Diffusive direct- and reflected field and airwave (ee-fields
        only).
      - 'dtetm' : as dsplit, but direct fielt TE, TM; reflected field TE, TM,
        and airwave (ee-fields only).

    signal : {None, 0, 1, -1}, default: None
        Source signal:

        - None: Frequency-domain response
        - -1 : Switch-off time-domain response
        - 0 : Impulse time-domain response
        - +1 : Switch-on time-domain response

    ab : int, default: 11
        Source-receiver configuration.

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

    aniso : float, default: 1.0
        Anisotropy lambda = sqrt(rho_v/rho_h) (-).

    epermH, epermV : float, default: 1.0
        Relative horizontal/vertical electric permittivity
        epsilon_h/epsilon_v (-). If epermH is provided but not epermV,
        isotropic behaviour is assumed.
        These parameters are ignored for the diffusive solution.

    mpermH, mpermV : float, default: 1.0
        Relative horizontal/vertical magnetic permeability mu_h/mu_v (-);
        #mpermH = #mpermV = #res. If mpermH is provided but not mpermV,
        isotropic behaviour is assumed.
        These parameters are ignored for the diffusive solution.

    verb : {0, 1, 2, 3, 4}, default: 2
        Level of verbosity:

        - 0: Print nothing.
        - 1: Print warnings.
        - 2: Print additional runtime
        - 3: Print additional start/stop, condensed parameter information.
        - 4: Print additional full parameter information

    squeeze : bool, default: True
        If True, the output is squeezed. If False, the output will always be of
        ``ndim=3``, (nfreqtime, nrec, nsrc).


    Returns
    -------
    EM : EMArray, (nfreqtime, nrec, nsrc)
        Frequency- or time-domain EM field (depending on `signal`):

        - If rec is electric, returns E [V/m].
        - If rec is magnetic, returns H [A/m].

        EMArray is a subclassed ndarray with `.pha` and `.amp` attributes
        (only relevant for frequency-domain data).

        The shape of EM is (nfreqtime, nrec, nsrc). However, single dimensions
        are removed.

        If `solution='dsplit'`, three ndarrays are returned: direct, reflect,
        air.

        If `solution='dtetm'`, five ndarrays are returned: direct_TE,
        direct_TM, reflect_TE, reflect_TM, air.


    Examples
    --------

    .. ipython::

       In [1]: import empymod
          ...: import numpy as np
          ...: src = [0, 0, 0]
          ...: rec = [np.arange(1, 11)*500, np.zeros(10), 200]
          ...: res = 50
          ...: EMfield = empymod.analytical(src, rec, res, freqtime=1, verb=0)
          ...: EMfield[0]
       Out[1]: (4.030914049602561e-08-9.691638183648923e-10j)

    """
    # Get kwargs with defaults.
    verb, squeeze = get_kwargs(['verb', 'squeeze'], [2, True], kwargs)

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

    # Reshape for number of sources
    if solution[1:] == 'split':
        EM = (EM[0].reshape((-1, nrec, nsrc), order='F'),
              EM[1].reshape((-1, nrec, nsrc), order='F'),
              EM[2].reshape((-1, nrec, nsrc), order='F'))
    elif solution[1:] == 'tetm':
        EM = (EM[0].reshape((-1, nrec, nsrc), order='F'),
              EM[1].reshape((-1, nrec, nsrc), order='F'),
              EM[2].reshape((-1, nrec, nsrc), order='F'),
              EM[3].reshape((-1, nrec, nsrc), order='F'),
              EM[4].reshape((-1, nrec, nsrc), order='F'))
    else:
        EM = EM.reshape((-1, nrec, nsrc), order='F')
    if squeeze:
        EM = np.squeeze(EM)

    # === 4.  FINISHED ============
    printstartfinish(verb, t0)

    return EMArray(EM)


def gpr(src, rec, depth, res, freqtime, cf, gain=None, ab=11, aniso=None,
        epermH=None, epermV=None, mpermH=None, mpermV=None, **kwargs):
    r"""Return Ground-Penetrating Radar signal.

    THIS FUNCTION IS EXPERIMENTAL, USE WITH CAUTION.

    It is rather an example how you can calculate GPR responses; however, DO
    NOT RELY ON IT! It works only well with QUAD or QWE (`quad`, `qwe`) for
    the Hankel transform, and with FFT (`fft`) for the Fourier transform.

    It calls internally :func:`dipole` for the frequency-domain calculation. It
    subsequently convolves the response with a Ricker wavelet with central
    frequency `cf`. If signal!=None, it carries out the Fourier transform and
    applies a gain to the response.


    Parameters
    ----------
    src, rec, freqtime : survey parameters
        See docstring of :func:`dipole` for a description.

    depth, res, aniso, epermH, epermV, mpermH, mpermV : model parameters
        See docstring of :func:`dipole` for a description.

    cf : float
        Centre frequency of GPR-signal, in Hz. Sensible values are between
        10 MHz and 3000 MHz.

    gain : float
        Power of gain function. If None, no gain is applied. Only used if
        signal!=None.

    ht, htarg, ft, ftarg, xdirect, loop : settings, optinal
        See docstring of :func:`bipole` for a description.


    Returns
    -------
    EM : ndarray
        GPR response

    """
    # Get kwargs with defaults.
    out = get_kwargs(['verb', 'ht', 'htarg', 'ft', 'ftarg', 'xdirect', 'loop'],
                     [2, 'quad', {}, 'fft', {}, False, None], kwargs)
    verb, ht, htarg, ft, ftarg, xdirect, loop = out

    if verb > 2:
        print("   GPR             :  EXPERIMENTAL, USE WITH CAUTION")
        print(f"     > centre freq :  {cf}")
        print(f"     > gain        :  {gain}")

    # === 1.  CHECK TIME ============

    # Check times and Fourier Transform arguments, get required frequencies
    time, freq, ft, ftarg = check_time(freqtime, 0, ft, ftarg, verb)

    # === 2. CALL DIPOLE ============

    EM = dipole(src=src, rec=rec, depth=depth, res=res, freqtime=freq, ab=ab,
                aniso=aniso, epermH=epermH, epermV=epermV, mpermH=mpermH,
                mpermV=mpermV, xdirect=xdirect, ht=ht, htarg=htarg, ft=ft,
                ftarg=ftarg, loop=loop, verb=verb)

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
             epermH=None, epermV=None, mpermH=None, mpermV=None, **kwargs):
    r"""Return electromagnetic wavenumber-domain field.

    Calculate the electromagnetic wavenumber-domain field due to infinitesimal
    small electric or magnetic dipole source(s), measured by infinitesimal
    small electric or magnetic dipole receiver(s); sources and receivers are
    directed along the principal directions x, y, or z, and all sources are at
    the same depth, as well as all receivers are at the same depth.


    See Also
    --------
    :func:`dipole` : EM fields due to infinitesimal small EM dipoles.
    :func:`bipole` : EM fields due to arbitrary rotated, finite length EM
                     dipoles.
    :func:`loop` : EM fields due to a magnetic source loop.


    Parameters
    ----------
    src, rec : list of floats or arrays
        Source and receiver coordinates [x, y, z] (m):

        - The x- and y-coordinates can be arrays, z is a single value.
        - The x- and y-coordinates must have the same dimension.
        - The x- and y-coordinates only matter for the angle-dependent factor.

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

    ab : int, default: 11
        Source-receiver configuration.

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

    aniso : array_like, default: ones
        Anisotropies lambda = sqrt(rho_v/rho_h) (-); #aniso = #res.

    epermH, epermV : array_like, default: ones
        Relative horizontal/vertical electric permittivities
        epsilon_h/epsilon_v (-); #epermH = #epermV = #res. If epermH is
        provided but not epermV, isotropic behaviour is assumed.

    mpermH, mpermV : array_like, default: ones
        Relative horizontal/vertical magnetic permeabilities mu_h/mu_v (-);
        #mpermH = #mpermV = #res. If mpermH is provided but not mpermV,
        isotropic behaviour is assumed.

    verb : {0, 1, 2, 3, 4}, default: 2
        Level of verbosity:

        - 0: Print nothing.
        - 1: Print warnings.
        - 2: Print additional runtime and kernel calls
        - 3: Print additional start/stop, condensed parameter information.
        - 4: Print additional full parameter information


    Returns
    -------
    PJ0, PJ1 : array
        Wavenumber-domain EM responses:

        - PJ0: Wavenumber-domain solution for the kernel with a Bessel function
          of the first kind of order zero.
        - PJ1: Wavenumber-domain solution for the kernel with a Bessel function
          of the first kind of order one.


    Examples
    --------

    .. ipython::

       In [1]: import empymod
          ...: import numpy as np
          ...: src = [0, 0, 100]
          ...: rec = [5000, 0, 200]
          ...: depth = [0, 300, 1000, 1050]
          ...: res = [1e20, .3, 1, 50, 1]
          ...: freq = 1
          ...: wavenr = np.logspace(-3.7, -3.6, 10)
          ...: PJ0, PJ1 = empymod.dipole_k(
          ...:         src, rec, depth, res, freq, wavenr, verb=0)
          ...: PJ0[0]
       Out[1]: (-2.5768974970445326e-08-2.0489943182087426e-09j)

       In [2]: PJ1[0]
       Out[2]: (1.9050482781619523e-10-6.842938067042929e-10j)

    """
    # Get kwargs with defaults.
    verb = get_kwargs(['verb', ], [2, ], kwargs)[0]

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
    ang_fact = kernel.angle_factor(angle, ab, msrc, mrec)

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
                                        False, msrc, mrec)

        # Collect output
        if J1 is not None:
            PJ1 += ang_fact[:, np.newaxis]*J1
            if ab in [11, 12, 21, 22, 14, 24, 15, 25]:  # Because of J2
                # J2(kr) = 2/(kr)*J1(kr) - J0(kr)
                PJ1 /= off[:, None]
        if J0 is not None:
            PJ0 += J0
        if J0b is not None:
            PJ0 += ang_fact[:, np.newaxis]*J0b

    # === 4.  FINISHED ============
    printstartfinish(verb, t0, 1)

    return np.squeeze(PJ0), np.squeeze(PJ1)


# Core modelling routines

def fem(ab, off, angle, zsrc, zrec, lsrc, lrec, depth, freq, etaH, etaV, zetaH,
        zetaV, xdirect, isfullspace, ht, htarg, msrc, mrec, loop_freq,
        loop_off, conv=True):
    r"""Return electromagnetic frequency-domain response.

    This function is called from one of the modelling routines
    :mod:`empymod.model`. Consult those for more details regarding the input
    and output parameters.

    This function can be used directly if you are sure the provided input is in
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
        ang_fact = kernel.angle_factor(angle, ab, msrc, mrec)

        calc = getattr(transform, 'hankel_'+ht)
        if loop_freq:

            for i in range(freq.size):
                out = calc(zsrc, zrec, lsrc, lrec, off, ang_fact, depth, ab,
                           etaH[None, i, :], etaV[None, i, :],
                           zetaH[None, i, :], zetaV[None, i, :], xdir,
                           htarg, msrc, mrec)
                fEM[None, i, :] += out[0]
                kcount += out[1]
                conv *= out[2]

        elif loop_off:
            for i in range(off.size):

                out = calc(zsrc, zrec, lsrc, lrec, off[None, i],
                           ang_fact[None, i], depth, ab, etaH, etaV, zetaH,
                           zetaV, xdir, htarg, msrc, mrec)
                fEM[:, None, i] += out[0]
                kcount += out[1]
                conv *= out[2]
        else:
            out = calc(zsrc, zrec, lsrc, lrec, off, ang_fact, depth, ab, etaH,
                       etaV, zetaH, zetaV, xdir, htarg, msrc, mrec)
            fEM += out[0]
            kcount += out[1]
            conv *= out[2]

    return fEM, kcount, conv


def tem(fEM, off, freq, time, signal, ft, ftarg, conv=True):
    r"""Return time-domain response of the frequency-domain response fEM.

    This function is called from one of the modelling routines
    :mod:`empymod.model`. Consult those for more details regarding the input
    and output parameters.

    This function can be used directly if you are sure the provided input is in
    the correct format. This is useful for inversion routines and similar, as
    it can speed-up the calculation by omitting input-checks.

    """
    # 1. Scale frequencies if switch-on/off response
    # Step function for causal times is like a unit fct, therefore an impulse
    # in frequency domain
    if signal in [-1, 1]:
        # Divide by signal/(2j*pi*f) to obtain step response
        fact = signal/(2j*np.pi*freq).ravel()
    else:
        fact = 1

    # 2. f->t transform
    calc = getattr(transform, 'fourier_'+ft)
    tEM = np.zeros((time.size, off.size))
    for i in range(off.size):
        out = calc(fEM[:, i]*fact, time, freq, ftarg)
        tEM[:, i] += out[0]
        conv *= out[1]

    return tEM*2/np.pi, conv  # Scaling from Fourier transform
