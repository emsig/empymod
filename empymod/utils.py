"""

:mod:`utils` -- Utilites
========================

This module consists of four groups of functions:
   0. General Settings
   1. Class EMArray
   2. Input parameter checks for modelling
   3. General utilities

Group 0 is to set minimum offset, frequency and time for calculation (in order
to avoid divisions by zero).  Group 2 are checks organised in modules. So if
you create for instance a modelling-routine in which you loop over frequencies,
you have to call `check_ab`, `check_param`, `check_spatial`, and `check_hankel`
only once, but `check_frequency` in each loop. You do not have to run these
checks if you are sure your input parameters are in the correct format.

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
from datetime import timedelta
from timeit import default_timer
from scipy import special
from scipy.constants import mu_0       # Magn. permeability of free space [H/m]
from scipy.constants import epsilon_0  # Elec. permittivity of free space [F/m]

from . import filters, transform


__all__ = ['EMArray', 'fem_input', 'tem_input', 'check_ab', 'check_param',
           'check_spatial', 'check_hankel', 'check_frequency', 'check_opt',
           'check_time', 'printstartfinish', ]

# 0. Settings

min_off = 1e-3    # Minimum offset     [m]
min_freq = 1e-20  # Minimum frequency  [Hz]
min_time = 1e-20  # Minimum time       [s]


# 1. Class EMArray

class EMArray(np.ndarray):
    """Subclassing an ndarray: add *Amplitude* <amp> and *Phase* <pha>.

    Parameters
    ----------
    realpart : array
        1. Real part of input, if input is real or complex.
        2. Imaginary part of input, if input is pure imaginary.
        3. Complex input.

        In cases 2 and 3, `imagpart` must be None.

    imagpart: array, optional
        Imaginary part of input. Defaults to None.

    Attributes
    ----------
    amp : ndarray
        Amplitude of the input data.
    pha : ndarray
        Phase of the input data, in degrees, lag-defined (increasing with
        increasing offset.) To get lead-defined phases, multiply `imagpart` by
        -1 before passing through this function.

    Examples
    --------
    >>> import numpy as np
    >>> from empymod.utils import EMArray
    >>> emvalues = EMArray(np.array([1,2,3]), np.array([1, 0, -1]))
    >>> print('Amplitude : ', emvalues.amp)
    Amplitude :  [ 1.41421356  2.          3.16227766]
    >>> print('Phase     : ', emvalues.pha)
    Phase     :  [ 45.           0.         -18.43494882]

    """

    def __new__(cls, realpart, imagpart=None):
        """Create a new EMArray."""

        # Create complex obj
        if np.any(imagpart):
            obj = np.real(realpart) + 1j*np.real(imagpart)
        else:
            obj = np.asarray(realpart, dtype=complex)

        # Ensure its at least a 1D-Array, view cls
        obj = np.atleast_1d(obj).view(cls)

        # Store amplitude
        obj.amp = np.abs(obj)

        # Calculate phase, unwrap it, transform to degrees
        obj.pha = np.rad2deg(np.unwrap(np.angle(obj.real + 1j*obj.imag)))

        return obj


# 2. Input parameter checks for modelling

def fem_input(src, rec, depth, res, freq, ab, aniso, epermH, epermV, mpermH,
              mpermV, xdirect, ht, htarg, opt, loop, verb):
    """Provide correct input for frequency-domain EM calculation.

    This check-function is called from one of the modelling routines in
    :mod:`model`.  Consult these modelling routines for a description of the
    input parameters.

    Returns
    -------
    outdata : tuple
        Tuple containing the correct input format for :mod:`model.fem`.

    """
    # Check src-rec configuration
    # => Get flags if src or rec or both are magnetic (msrc, mrec)
    ab_calc, msrc, mrec = check_ab(ab, verb)

    # Check layer parameters
    param = check_param(depth, res, aniso, epermH, epermV, mpermH, mpermV,
                        verb)
    depth, res, aniso, epermH, epermV, mpermH, mpermV, isfullspace = param

    # Check src and rec
    # => Get source and receiver depths (zsrc, zrec)
    # => Get layer number in which src and rec reside (lsrc/lrec)
    # => Get offsets and angles (off, angle)
    spatial = check_spatial(src, rec, depth, verb)
    zsrc, zrec, lsrc, lrec, off, angle = spatial

    # Check Hankel transform parameters
    ht, htarg = check_hankel(ht, htarg, ab, verb)

    # Check frequency
    # => Get etaH, etaV, zetaH, and zetaV
    frequency = check_frequency(freq, res, aniso, epermH, epermV, mpermH,
                                mpermV, verb)
    freq, etaH, etaV, zetaH, zetaV = frequency

    # Check optimization
    optimization = check_opt(opt, off, freq, loop, ht, htarg, verb)
    use_spline, use_ne_eval, loop_freq, loop_off = optimization

    # Print calculation related info
    if verb > 1:
        if ab_calc in [36, ]:
            print("\n>  <ab> IS "+str(ab_calc)+" WHICH IS ZERO; returning")

        elif isfullspace:
            print("\n>  MODEL IS A FULLSPACE; returning analytical " +
                  "frequency-domain solution")

        elif not isfullspace:
            print("\n>  CALCULATING MODEL")

    # Arrange outdata-tuple
    outdata = (ab_calc, off, angle, zsrc, zrec, lsrc, lrec, depth, freq, etaH,
               etaV, zetaH, zetaV, xdirect, isfullspace, ht, htarg, use_spline,
               use_ne_eval, msrc, mrec, loop_freq, loop_off)

    return outdata


def tem_input(src, rec, depth, res, time, ab, signal, aniso, epermH, epermV,
              mpermH, mpermV, xdirect, ht, htarg, ft, ftarg, opt, loop, verb):
    """Provide correct input for time-domain EM calculation.

    This check-function is called from one of the modelling routines in
    :mod:`model`.  Consult these modelling routines for a description of the
    input parameters.

    Returns
    -------
    outdata : tuple
        Tuple containing the correct input format for :mod:`model.tem`.

    """
    # Check times and Fourier Transform arguments, get required frequencies
    time, signal, freq, ft, ftarg = check_time(time, signal, ft, ftarg, verb)

    # Check the normal frequency-calculation stuff
    fdata = fem_input(src, rec, depth, res, freq, ab, aniso, epermH, epermV,
                      mpermH, mpermV, xdirect, ht, htarg, opt, loop, verb)

    # If verbose, indicate f->t transform
    if verb > 1:
        print("\n>  f->t TRANSFORM")

    # Arrange outdata-tuple
    outdata = fdata[:17] + (time, signal, ft, ftarg) + fdata[17:]

    return outdata


def check_ab(ab, verb):
    """Check source-receiver configuration.

    This check-function is called from one of the modelling routines in
    :mod:`model`.  Consult these modelling routines for a description of the
    input parameters.

    Returns
    -------
    ab_calc : int
        Adjusted configuration using reciprocity.
    msrc : bool
        If True, src is magnetic; if False, src is electric.
    mrec : bool
        If True, rec is magnetic; if False, rec is electric.

    """

    # Try to cast ab into an integer
    try:
        int(ab)
    except(TypeError, ValueError):
        print('* ERROR   :: <ab> must be an integer')
        raise

    # Check src and rec orientation (<ab> for alpha-beta)
    # pab: all possible values that <ab> can take
    pab = [11, 12, 13, 14, 15, 16, 21, 22, 23, 24, 25, 26,
           31, 32, 33, 34, 35, 36, 41, 42, 43, 44, 45, 46,
           51, 52, 53, 54, 55, 56, 61, 62, 63, 64, 65, 66]
    if ab not in pab:
        print('* ERROR   :: <ab> must be one of: ' + str(pab) + ';' +
              ' <ab> provided: ' + str(ab))
        raise ValueError('ab')

    # Print input <ab>
    if verb > 1:
        print("   Input ab    : ", ab)

    # Check if src and rec are magnetic or electric
    msrc = ab % 10 > 3   # If True: magnetic src
    mrec = ab // 10 > 3  # If True: magnetic rec

    # If rec is magnetic, switch <ab> using reciprocity.
    if mrec:
        if msrc:
            # G^mm_ab(s, r, e, z) = -G^ee_ab(s, r, -z, -e)
            ab_calc = ab-33  # -30 : mrec->erec; -3: msrc->esrc
        else:
            # G^me_ab(s, r, e, z) = -G^em_ba(r, s, e, z)
            ab_calc = ab % 10*10 + ab // 10  # Swap alpha/beta
    else:
        ab_calc = ab

    # Print actual calculated <ab>
    if verb > 1:
        print("   Calc. ab    : ", ab_calc)

    return ab_calc, msrc, mrec


def check_param(depth, res, aniso, epermH, epermV, mpermH, mpermV, verb):
    """Check depth and corresponding layer parameters.

    This check-function is called from one of the modelling routines in
    :mod:`model`.  Consult these modelling routines for a description of the
    input parameters.

    Returns
    -------
    depth : array
        Depths of layer interfaces, adds -infty at beginning if not present.
    res : array
        Resistivities as provided, checked for size.
    aniso, epermH, epermV, mpermH, mpermV : array
        Parameters as provided, checked for size. If None provided, defaults to
        an array of ones.
    isfullspace : bool
        If True, the model is a fullspace (res, aniso, epermH, epermV, mpermM,
        and mpermV are in all layers the same).

    """

    # Check depth
    depth = _check_var(depth, float, 1, 'depth')

    # Add -infinity at the beginning
    # => The top-layer (-infinity to first interface) is layer 0.
    depth = np.insert(depth, 0, -np.infty)

    # Ensure depth is increasing
    if np.any(depth[1:] - depth[:-1] < 0):
        print('* ERROR   :: <depth> must be increasing;' +
              ' <depth> provided: ' + _strvar(depth))
        raise ValueError('ab')

    # Cast and check resistivity
    res = _check_var(res, float, 1, 'res', depth.shape)

    # Check anisotropy, electric permittivity, and magnetic permeability
    def check_inp(var, name):
        """Param-check function. Default to ones if not provided"""
        if not np.any(var):
            return np.ones(depth.size)
        else:
            return _check_var(var, float, 1, name, depth.shape)

    aniso = check_inp(aniso, 'aniso')
    epermH = check_inp(epermH, 'epermH')
    epermV = check_inp(epermV, 'epermV')
    mpermH = check_inp(mpermH, 'mpermH')
    mpermV = check_inp(mpermV, 'mpermV')

    # Print model parameters
    if verb > 1:
        print("   depth   [m] : ", _strvar(depth[1:]))
        print("   res [Ohm.m] : ", _strvar(res))
        print("   aniso   [-] : ", _strvar(aniso))
        print("   epermH  [-] : ", _strvar(epermH))
        print("   epermV  [-] : ", _strvar(epermV))
        print("   mpermH  [-] : ", _strvar(mpermH))
        print("   mpermV  [-] : ", _strvar(mpermV))

    # Check if medium is a homogeneous full-space. If that is the case, the
    # EM-field is computed analytically directly in the frequency-domain.
    # Note: Also a stack of layers with the same material parameters is treated
    #       as a homogeneous full-space.
    if res.shape == ():
        isfullspace = True
    else:
        isores = (res - res[0] == 0).all()*(aniso - aniso[0] == 0).all()
        isoeph = (epermH - epermH[0] == 0).all()
        isoepv = (epermV - epermV[0] == 0).all()
        isomph = (mpermH - mpermH[0] == 0).all()
        isompv = (mpermV - mpermV[0] == 0).all()
        isfullspace = isores*isoeph*isoepv*isomph*isompv

    return depth, res, aniso, epermH, epermV, mpermH, mpermV, isfullspace


def check_spatial(src, rec, depth, verb):
    """Check spatial input parameters.

    This check-function is called from one of the modelling routines in
    :mod:`model`.  Consult these modelling routines for a description of the
    input parameters.

    Returns
    -------
    zsrc : float
        Depth of src.
    zrec : float
        Depth of rec.
    lsrc : int
        Layer number in which src resides.
    lrec : int
        Layer number in which rec resides.
    off : array of floats
        Offsets
    angle : array of floats
        Angles

    """

    # Check src
    _check_shape(np.squeeze(src), 'src', (3,))
    src[0] = _check_var(src[0], float, 1, 'src-x', (1,))
    src[1] = _check_var(src[1], float, 1, 'src-y', (1,))
    src[2] = _check_var(src[2], float, 0, 'src-z', ())
    zsrc = np.squeeze(src[2])

    # Check rec
    _check_shape(np.squeeze(rec), 'rec', (3,))
    rec[0] = _check_var(rec[0], float, 1, 'rec-x')
    rec[1] = _check_var(rec[1], float, 1, 'rec-y', rec[0].shape)
    rec[2] = _check_var(rec[2], float, 0, 'rec-z', ())
    zrec = np.squeeze(rec[2])

    # Determine layers in which src and rec reside.
    # Note: If src[2] or rec[2] are on a layer interface, the layer above the
    #       interface is chosen.
    depthinfty = np.concatenate((depth[1:], np.array([np.infty])))
    lsrc = np.where((depth < src[2])*(depthinfty >= src[2]))[0][0]
    lrec = np.where((depth < rec[2])*(depthinfty >= rec[2]))[0][0]

    # Coordinates
    xco = rec[0] - src[0]             # X-coordinates  [m]
    yco = rec[1] - src[1]             # Y-coordinates  [m]
    off = np.sqrt(xco*xco + yco*yco)  # Offset         [m]
    angle = np.arctan2(yco, xco)      # Angle        [rad]

    # Minimum offset to avoid singularities at off = 0 m.
    # => min_off is defined at the start of this file
    ioff = np.where(off < min_off)
    off[ioff] = min_off
    angle[ioff] = np.nan
    if np.size(ioff) != 0 and verb > 0:
        print('* WARNING :: Offsets <', min_off, 'm are set to', min_off, 'm!')

    # Print spatial parameters
    if verb > 1:
        print("   src x   [m] : ", _strvar(src[0]))
        print("   src y   [m] : ", _strvar(src[1]))
        print("   src z   [m] : ", _strvar(src[2]))
        print("   rec x   [m] : ", str(rec[0].min()), "-", str(rec[0].max()),
              ";", str(rec[0].size), " [min-max; #]")
        if verb > 2:
            print("               : ", _strvar(rec[0]))
        print("   rec y   [m] : ", str(rec[1].min()), "-", str(rec[1].max()),
              ";", str(rec[1].size), " [min-max; #]")
        if verb > 2:
            print("               : ", _strvar(rec[1]))
        print("   rec z   [m] : ", _strvar(rec[2]))

    return zsrc, zrec, lsrc, lrec, off, angle


def check_hankel(ht, htarg, ab, verb):
    """Check Hankel transform parameters.

    This check-function is called from one of the modelling routines in
    :mod:`model`.  Consult these modelling routines for a description of the
    input parameters.

    Returns
    -------
    ht, htarg
        Checked if valid and set to defaults if not provided.

    """

    # Ensure ht is all lowercase
    ht = ht.lower()

    if ht == 'fht':    # If FHT, check filter settings

        # Check Input
        if not htarg:  # If None, create empty list
            htarg = []
        elif not isinstance(htarg, (list, tuple)):  # If only filter
            htarg = [htarg, ]

        # Check filter; defaults to key_401_2009
        try:
            fhtfilt = htarg[0]
        except:
            fhtfilt = filters.key_401_2009()
        else:
            # If not already filter-instance, get it from string
            if not hasattr(fhtfilt, 'base'):
                fhtfilt = getattr(filters, fhtfilt)()

        # Check pts_per_dec; defaults to None
        try:
            pts_per_dec = htarg[1]
        except:
            pts_per_dec = None
        else:
            if pts_per_dec:  # Check pts_per_dec
                pts_per_dec = _check_var(pts_per_dec, int, 0,
                                         'fht: pts_per_dec', ())

        # Assemble htarg
        htarg = (fhtfilt, pts_per_dec)

        # If verbose, print Hankel transform information
        if verb > 1:
            print("   Hankel      :  Fast Hankel Transform")
            print("     > Filter  :  " + fhtfilt.name)

    elif ht in ['qwe']:
        # Rename ht
        ht = 'hqwe'

        # Get and check input or set defaults
        if not htarg:
            htarg = []

        # rtol : 1e-12 is low for accurate results
        try:
            rtol = _check_var(htarg[0], float, 0, 'qwe: rtol', ())
        except:
            rtol = float(1e-12)

        # atol : 1e-30 is low for accurate results
        try:
            atol = _check_var(htarg[1], float, 0, 'qwe: atol', ())
        except:
            atol = float(1e-30)

        # nquad : 51 is relatively high
        try:
            nquad = _check_var(htarg[2], int, 0, 'qwe: nquad', ())
        except:
            nquad = int(51)

        # maxint :  40/100 is relatively high
        #             40 : 11-15, 21-25, 33-35, 41-45, 51-55
        #            100 : 16/26, 31/32, 46/56, 61-66
        try:
            maxint = _check_var(htarg[3], int, 0, 'qwe: maxint', ())
        except:
            if ab in [16, 26, 31, 32, 46, 56, 61, 62, 64, 65, 66]:
                maxint = int(100)
            else:
                maxint = int(40)

        # pts_per_dec : 80 is relatively high
        try:
            pts_per_dec = _check_var(htarg[4], int, 0, 'qwe: pts_per_dec', ())
        except:
            pts_per_dec = int(80)

        # Assemble htarg
        htarg = (rtol, atol, nquad, maxint, pts_per_dec)

        # If verbose, print Hankel transform information
        if verb > 1:
            print("   Hankel      :  Quadrature-with-Extrapolation")
            print("     > rtol    :  " + str(htarg[0]))
            print("     > atol    :  " + str(htarg[1]))
            print("     > nquad   :  " + str(htarg[2]))
            print("     > maxint  :  " + str(htarg[3]))

    else:
        print("* ERROR   :: <ht> must be one of: ['fht', 'qwe'];" +
              " <ht> provided: " + str(ht))
        raise ValueError('ht')

    return ht, htarg


def check_frequency(freq, res, aniso, epermH, epermV, mpermH, mpermV, verb):
    """Calculate frequency-dependent parameters.

    This check-function is called from one of the modelling routines in
    :mod:`model`.  Consult these modelling routines for a description of the
    input parameters.

    Returns
    -------
    freq : float
        Frequency, checked for size and assured min_freq.
    etaH, etaV, zetaH, zetaV : array
        Parameters etaH, etaV, zetaH, and zetaV, same size as provided
        resistivity.

    """

    # Check frequency
    freq = _check_var(freq, float, 1, 'freq')

    # Minimum frequency to avoid division by zero at freq = 0 Hz.
    # => min_freq is defined at the start of this file
    ifreq = np.where(freq < min_freq)
    freq[ifreq] = min_freq
    if np.size(ifreq) != 0 and verb > 0:
        print('* WARNING :: Frequencies <', min_freq, 'Hz are set to',
              min_freq, 'Hz!')
    if verb > 1:
        print("   freq   [Hz] : ", str(freq.min()), "-", str(freq.max()), ";",
              str(freq.size), " [min-max; #]")
        if verb > 2:
            print("               : ", _strvar(freq))

    # Calculate eta and zeta (horizontal and vertical)
    etaH = 1/res + np.outer(2j*np.pi*freq, epermH*epsilon_0)
    etaV = 1/(res*aniso*aniso) + np.outer(2j*np.pi*freq, epermV*epsilon_0)
    zetaH = np.outer(2j*np.pi*freq, mpermH*mu_0)
    zetaV = np.outer(2j*np.pi*freq, mpermV*mu_0)

    return freq, etaH, etaV, zetaH, zetaV


def check_opt(opt, off, freq, loop, ht, htarg, verb):
    """Check optimization parameters.

    This check-function is called from one of the modelling routines in
    :mod:`model`.  Consult these modelling routines for a description of the
    input parameters.

    Returns
    -------
    use_spline : bool
        Boolean if to use spline interpolation.
    use_ne_eval : bool
        Boolean if to use `numexpr`.
    loop_freq : bool
        Boolean if to loop over frequencies.
    loop_off : bool
        Boolean if to loop over offsets.

    """
    # Check optimization flag
    if opt == 'spline':
        use_spline, use_ne_eval = True, False
    elif opt == 'parallel':
        use_spline, use_ne_eval = False, True
    else:
        use_spline, use_ne_eval = False, False

    # Define if to loop over frequencies or over offsets
    if ht == 'hqwe' or use_spline:
        loop_freq = True
        loop_off = False
    else:
        loop_off = loop == 'off'
        loop_freq = loop == 'freq'

    # If verbose, print optimization information
    if verb > 1:
        if use_spline:
            print("   Hankel Opt. :  Use spline")
            pstr = "     > pts/dec :  "
            if ht == 'hqwe':
                print(pstr + str(htarg[4]))
            else:
                if htarg[1]:
                    print(pstr + str(htarg[1]))
                else:
                    print(pstr + 'Defined by filter (lagged)')
        elif use_ne_eval:
            print("   Hankel Opt. :  Use parallel")
        else:
            print("   Hankel Opt. :  None")

        if loop_off:
            print("   Loop over   :  Offsets")
        elif loop_freq:
            print("   Loop over   :  Frequencies")
        else:
            print("   Loop over   :  None (all vectorized)")

    return use_spline, use_ne_eval, loop_freq, loop_off


def check_time(time, signal, ft, ftarg, verb):
    """Check time domain specific input parameters.

    This check-function is called from one of the modelling routines in
    :mod:`model`.  Consult these modelling routines for a description of the
    input parameters.

    Returns
    -------
    time : float
        Time, checked for size and assured min_time.
    freq : float
        Frequencies required for given times and ft-settings.
    ft, ftarg
        Checked if valid and set to defaults if not provided,
        checked with signal.

    """
    # Check time
    time = _check_var(time, float, 1, 'time')

    # Minimum time to avoid division by zero  at time = 0 s.
    # => min_time is defined at the start of this file
    itime = np.where(time < min_time)
    time[itime] = min_time
    if verb > 0 and np.size(itime) != 0:
        print('* WARNING :: Times <', min_time, 's are set to', min_time, 's!')
    if verb > 1:
        print("   time    [s] : ", str(time.min()), "-", str(time.max()), ";",
              str(time.size), " [min-max; #]")
        if verb > 2:
            print("               : ", _strvar(time))

    # Ensure ft is all lowercase
    ft = ft.lower()

    if ft in ['cos', 'sin']:    # If Cosine/Sine, check filter setting

        # If switch-off/on is required, ensure ft is sine
        # Sine-transform uses imaginary part, which is 0 at DC (-> late time)
        if signal != 0:
            ft = 'sin'

        # Check Input
        if not ftarg:  # If None, create empty list
            ftarg = []
        elif not isinstance(ftarg, (list, tuple)):  # If only filter
            ftarg = [ftarg, ]

        # Check filter; defaults to key_201_CosSin_2012
        try:
            fftfilt = ftarg[0]
        except:
            fftfilt = filters.key_201_CosSin_2012()
        else:
            # If not already filters-instance, get it from string
            if not hasattr(fftfilt, 'base'):
                fftfilt = getattr(filters, fftfilt)()

        # Check pts_per_dec; defaults to None
        try:
            pts_per_dec = ftarg[1]
        except:
            pts_per_dec = None
        else:
            if pts_per_dec:  # Check pts_per_dec
                pts_per_dec = _check_var(pts_per_dec, int, 0,
                                         ft + ' pts_per_dec', ())

        # Assemble ftarg
        ftarg = (fftfilt, ft, pts_per_dec)

        # If verbose, print Fourier transform information
        if verb > 1:
            if ft == 'sin':
                print("   Fourier     :  Sine-Filter")
            else:
                print("   Fourier     :  Cosine-Filter")
            print("     > Filter  :  " + ftarg[0].name)
            pstr = "     > pts/dec :  "
            if ftarg[2]:
                print(pstr + str(ftarg[2]))
            else:
                print(pstr + 'Defined by filter (lagged)')

        # Get required frequencies
        # (multiply time by 2Pi, as calculation is done in angular frequencies)
        freq, _ = transform.get_spline_values(ftarg[0], 2*np.pi*time, ftarg[2])
        freq = np.squeeze(freq)

        # Rename ft
        ft = 'fft'

    elif ft == 'qwe':    # QWE
        # Rename ft
        ft = 'fqwe'

        # Get and check input or set defaults
        if not ftarg:  # Default values
            ftarg = []

        try:  # rtol
            rtol = _check_var(ftarg[0], float, 0, 'qwe: rtol', ())
        except:
            rtol = float(1e-8)

        try:  # atol
            atol = _check_var(ftarg[1], float, 0, 'qwe: atol', ())
        except:
            atol = float(1e-20)

        try:  # nquad
            nquad = _check_var(ftarg[2], int, 0, 'qwe: nquad', ())
        except:
            nquad = int(21)

        try:  # maxint
            maxint = _check_var(ftarg[3], int, 0, 'qwe: maxint', ())
        except:
            maxint = int(200)

        try:  # pts_per_dec
            pts_per_dec = _check_var(ftarg[4], int, 0, 'qwe: pts_per_dec', ())
        except:
            pts_per_dec = int(20)

        # Assemble ftarg
        ftarg = (rtol, atol, nquad, maxint, pts_per_dec)

        # If verbose, print Fourier transform information
        if verb > 1:
            print("   Fourier      :  Quadrature-with-Extrapolation")
            print("     > rtol    :  " + str(ftarg[0]))
            print("     > atol    :  " + str(ftarg[1]))
            print("     > nquad   :  " + str(ftarg[2]))
            print("     > maxint  :  " + str(ftarg[3]))
            print("     > pts/dec :  " + str(ftarg[4]))

        # Get required frequencies
        g_x, _ = special.p_roots(ftarg[2])
        minf = np.floor(10*np.log10((g_x.min() + 1)*np.pi/2/time.max()))/10
        maxf = np.ceil(10*np.log10(ftarg[3]*np.pi/time.min()))/10
        freq = np.logspace(minf, maxf, (maxf-minf)*ftarg[4] + 1)

    elif ft == 'fftlog':    # FFTLog

        # Get and check input or set defaults
        if not ftarg:  # Default values
            ftarg = []

        try:  # pts_per_dec
            pts_per_dec = _check_var(ftarg[0], int, 0,
                                     'fftlog: pts_per_dec', ())
        except:
            pts_per_dec = 10

        try:  # add_dec
            add_dec = _check_var(ftarg[1], float, 1, 'fftlog: add_dec', (2,))
        except:
            add_dec = np.array([-2, 1])

        try:  # q
            q = _check_var(ftarg[2], float, 0, 'fftlog: q', ())
            # Restrict q to +/- 1
            if np.abs(q) > 1:
                q = np.sign(q)
        except:
            q = np.array(0)

        # If verbose, print Fourier transform information
        if verb > 1:
            print("   Fourier      :  FFTLog ")
            print("     > pts/dec  :  " + str(pts_per_dec))
            print("     > add_dec  :  " + str(add_dec))
            print("     > q        :  " + str(q))

        # Calculate minimum and maximum required frequency
        minf = np.log10(1/time.max()) + add_dec[0]
        maxf = np.log10(1/time.min()) + add_dec[1]
        n = np.int(maxf - minf)*pts_per_dec

        # Initialize FFTLog, get required parameters
        freq, tcalc, dlnr, kr, rk = transform.fhti(minf, maxf, n, q)

        # Assemble ftarg
        ftarg = (tcalc, dlnr, kr, rk, q)

    else:
        print("* ERROR   :: <ft> must be one of: ['cos', 'sin', 'qwe', " +
              "'fftlog']; <ft> provided: "+str(ft))
        raise ValueError('ft')

    return time, signal, freq, ft, ftarg


# 3. Internal utilities

# TODO : ab to msrc/mrec/xyz/xyz

def _strvar(a, prec='{:G}'):
    """Return variable as a string to print, with given precision."""
    return ' '.join([prec.format(i) for i in np.atleast_1d(a)])


def _check_var(var, dtype, ndmin, name, shape=None):
    var = np.array(var, dtype=dtype, copy=True, ndmin=ndmin)
    if shape:
        _check_shape(var, name, shape)
    return var


def _check_shape(var, name, shape):
    """Check that <var> has shape <shape>; if false raise ValueError(name)"""
    varshape = np.shape(var)
    if shape != varshape:
        print('* ERROR   :: Parameter ' + name + ' has wrong shape! : ' +
              str(varshape) + ' instead of ' + str(shape) + '.')
        raise ValueError(name)


def printstartfinish(verb, inp=None):
    """Print start and finish with time measure."""
    if inp:
        print('\n:: empymod END; runtime = ' +
              str(timedelta(seconds=default_timer() - inp)) + ' ::\n')
    else:
        t0 = default_timer()
        if verb > 1:
            print("\n:: empymod START  ::\n\n>  INPUT CHECK")
        return t0
