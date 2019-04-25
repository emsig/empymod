r"""

:mod:`utils` -- Utilites
========================

Utilities for ``model`` such as checking input parameters.

This module consists of four groups of functions:
   0. General settings
   1. Class EMArray
   2. Input parameter checks for modelling
   3. Internal utilities

"""
# Copyright 2016-2019 Dieter Werthmüller
#
# This file is part of empymod.
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


# Mandatory imports
import warnings
import numpy as np
from scipy import special
from datetime import timedelta
from timeit import default_timer


# Optional imports
try:
    import numexpr
    # Ensure Intel's Vector Math Library
    if not numexpr.use_vml:
        numexpr = False
        numexpr_msg = "* WARNING :: `numexpr` is not installed with VML, "
        numexpr_msg += "`opt=='parallel'` has no effect."
except ImportError:
    numexpr = False
    numexpr_msg = "* WARNING :: `numexpr` is not installed, "
    numexpr_msg += "`opt=='parallel'` has no effect."

# Relative imports
from . import filters, transform


__all__ = ['EMArray', 'check_time_only', 'check_time', 'check_model',
           'check_frequency', 'check_hankel', 'check_opt', 'check_dipole',
           'check_bipole', 'check_ab', 'check_solution', 'get_abs',
           'get_geo_fact', 'get_azm_dip', 'get_off_ang', 'get_layer_nr',
           'printstartfinish', 'conv_warning', 'set_minimum', 'get_minimum',
           'spline_backwards_hankel']

# 0. General settings

_min_freq = 1e-20   # Minimum frequency  [Hz]
_min_time = 1e-20   # Minimum time       [s]
_min_off = 1e-3     # Minimum offset     [m]
#                   # > Also used to round src- & rec-coordinates (1e-3 => mm)
_min_res = 1e-20    # Minimum value for horizontal/vertical resistivity
_min_angle = 1e-10  # Angle factors smaller than that are set to 0

# Define all errors we want to catch with the variable-checks and setting of
# default values. This is not perfect, but better than 'except Exception'.
VariableCatch = (LookupError, AttributeError, ValueError, TypeError, NameError)


# 1. Class EMArray

class EMArray(np.ndarray):
    r"""Subclassing an ndarray: add *amplitude* <amp> and *phase* <pha>.

    Parameters
    ----------
    realpart : array
        1. Real part of input, if input is real or complex.
        2. Imaginary part of input, if input is pure imaginary.
        3. Complex input.

        In cases 2 and 3, ``imagpart`` must be None.

    imagpart: array, optional
        Imaginary part of input. Defaults to None.

    Attributes
    ----------
    amp : ndarray
        Amplitude of the input data.

    pha : ndarray
        Phase of the input data, in degrees, lag-defined (increasing with
        increasing offset.) To get lead-defined phases, multiply ``imagpart``
        by -1 before passing through this function.

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
        r"""Create a new EMArray."""

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

# 2.a <Check>s (alphabetically)

def check_ab(ab, verb):
    r"""Check source-receiver configuration.

    This check-function is called from one of the modelling routines in
    :mod:`model`. Consult these modelling routines for a detailed description
    of the input parameters.

    Parameters
    ----------
    ab : int
        Source-receiver configuration.

    verb : {0, 1, 2, 3, 4}
        Level of verbosity.


    Returns
    -------
    ab_calc : int
        Adjusted source-receiver configuration using reciprocity.

    msrc, mrec : bool
        If True, src/rec is magnetic; if False, src/rec is electric.

    """

    # Try to cast ab into an integer
    try:
        ab = int(ab)
    except VariableCatch:
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
    if verb > 2:
        print("   Input ab        : ", ab)

    # Check if src and rec are magnetic or electric
    msrc = ab % 10 > 3   # If True: magnetic src
    mrec = ab // 10 > 3  # If True: magnetic rec

    # If rec is magnetic, switch <ab> using reciprocity.
    if mrec:
        if msrc:
            # G^mm_ab(s, r, e, z) = -G^ee_ab(s, r, -z, -e)
            ab_calc = ab - 33  # -30 : mrec->erec; -3: msrc->esrc
        else:
            # G^me_ab(s, r, e, z) = -G^em_ba(r, s, e, z)
            ab_calc = ab % 10*10 + ab // 10  # Swap alpha/beta
    else:
        ab_calc = ab

    # Print actual calculated <ab>
    if verb > 2:
        if ab in [36, 63]:
            print("\n>  <ab> IS "+str(ab)+" WHICH IS ZERO; returning")
        else:
            print("   Calculated ab   : ", ab_calc)

    return ab_calc, msrc, mrec


def check_bipole(inp, name):
    r"""Check di-/bipole parameters.

    This check-function is called from one of the modelling routines in
    :mod:`model`.  Consult these modelling routines for a detailed description
    of the input parameters.

    Parameters
    ----------
    inp : list of floats or arrays
        Coordinates of inp (m):
        [dipole-x, dipole-y, dipole-z, azimuth, dip] or.
        [bipole-x0, bipole-x1, bipole-y0, bipole-y1, bipole-z0, bipole-z1].

    name : str, {'src', 'rec'}
        Pole-type.

    Returns
    -------
    inp : list
        As input, checked for type and length.

    ninp : int
        Number of inp.

    ninpz : int
        Number of inp depths (ninpz is either 1 or ninp).

    isdipole : bool
        True if inp is a dipole.

    """

    def chck_dipole(inp, name):
        r"""Check inp for shape and type."""
        # Check x
        inp[0] = _check_var(inp[0], float, 1, name+'-x')

        # Check y and ensure it has same dimension as x
        inp[1] = _check_var(inp[1], float, 1, name+'-y', inp[0].shape)

        # Check z
        inp[2] = _check_var(inp[2], float, 1, name+'-z', (1,), inp[0].shape)

        # Check if all depths are the same, if so replace by one value
        if np.all(np.isclose(inp[2]-inp[2][0], 0)):
            inp[2] = np.array([inp[2][0]])

        return inp

    # Check length of inp.
    narr = len(inp)
    if narr not in [5, 6]:
        print('* ERROR   :: Parameter ' + name + ' has wrong length! : ' +
              str(narr) + ' instead of 5 (dipole) or 6 (bipole).')
        raise ValueError(name)

    # Flag if it is a dipole or not
    isdipole = narr == 5

    if isdipole:  # dipole checks
        # Check x, y, and z
        inp = chck_dipole(inp, name)

        # Check azimuth and dip (must be floats, otherwise use ``bipole``)
        inp[3] = _check_var(inp[3], float, 1, 'azimuth', (1,))
        inp[4] = _check_var(inp[4], float, 1, 'dip', (1,))

        # How many different depths
        inpz = inp[2].size

    else:         # bipole checks
        # Check each pole for x, y, and z
        inp0 = chck_dipole(inp[::2], name+'-1')   # [x0, y0, z0]
        inp1 = chck_dipole(inp[1::2], name+'-2')  # [x1, y1, z1]

        # If one pole has a single depth, but the other has various
        # depths, we have to repeat the single depth, as we will have
        # to loop over them.
        if inp0[2].size != inp1[2].size:
            if inp0[2].size == 1:
                inp0[2] = np.repeat(inp0[2], inp1[2].size)
            else:
                inp1[2] = np.repeat(inp1[2], inp0[2].size)

        # Check if inp is a dipole instead of a bipole
        # (This is a problem, as we would could not define the angles then.)
        if not np.all((inp0[0] != inp1[0]) + (inp0[1] != inp1[1]) +
                      (inp0[2] != inp1[2])):
            print("* ERROR   :: At least one of <" + name + "> is a point " +
                  "dipole, use the format [x, y, z, azimuth, dip] instead " +
                  "of [x0, x1, y0, y1, z0, z1].")
            raise ValueError('Bipole: bipole-' + name)

        # Collect elements
        inp = [inp0[0], inp1[0], inp0[1], inp1[1], inp0[2], inp1[2]]

        # How many different depths
        inpz = inp[4].size

    return inp, inp[0].size, inpz, isdipole


def check_dipole(inp, name, verb):
    r"""Check dipole parameters.

    This check-function is called from one of the modelling routines in
    :mod:`model`.  Consult these modelling routines for a detailed description
    of the input parameters.

    Parameters
    ----------
    inp : list of floats or arrays
        Pole coordinates (m): [pole-x, pole-y, pole-z].

    name : str, {'src', 'rec'}
        Pole-type.

    verb : {0, 1, 2, 3, 4}
        Level of verbosity.

    Returns
    -------
    inp : list
        List of pole coordinates [x, y, z].

    ninp : int
        Number of inp-elements

    """

    # Check inp for x, y, and z; x & y must have same length, z is a float
    _check_shape(np.squeeze(inp), name, (3,))
    inp[0] = _check_var(inp[0], float, 1, name+'-x')
    inp[1] = _check_var(inp[1], float, 1, name+'-y', inp[0].shape)
    inp[2] = _check_var(inp[2], float, 1, name+'-z', (1,))

    # Print spatial parameters
    if verb > 2:
        # Pole-type: src or rec
        if name == 'src':
            longname = '   Source(s)       : '
        else:
            longname = '   Receiver(s)     : '

        print(longname, str(inp[0].size), 'dipole(s)')
        tname = ['x  ', 'y  ', 'z  ']
        for i in range(3):
            text = "     > " + tname[i] + "     [m] : "
            _prnt_min_max_val(inp[i], text, verb)

    return inp, inp[0].size


def check_frequency(freq, res, aniso, epermH, epermV, mpermH, mpermV, verb):
    r"""Calculate frequency-dependent parameters.

    This check-function is called from one of the modelling routines in
    :mod:`model`.  Consult these modelling routines for a detailed description
    of the input parameters.

    Parameters
    ----------
    freq : array_like
        Frequencies f (Hz).

    res : array_like
        Horizontal resistivities rho_h (Ohm.m); #res = #depth + 1.

    aniso : array_like
        Anisotropies lambda = sqrt(rho_v/rho_h) (-); #aniso = #res.

    epermH, epermV : array_like
        Relative horizontal/vertical electric permittivities
        epsilon_h/epsilon_v (-);
        #epermH = #epermV = #res.

    mpermH, mpermV : array_like
        Relative horizontal/vertical magnetic permeabilities mu_h/mu_v (-);
        #mpermH = #mpermV = #res.

    verb : {0, 1, 2, 3, 4}
        Level of verbosity.


    Returns
    -------
    freq : float
        Frequency, checked for size and assured min_freq.

    etaH, etaV : array
        Parameters etaH/etaV, same size as provided resistivity.

    zetaH, zetaV : array
        Parameters zetaH/zetaV, same size as provided resistivity.

    """
    global _min_freq

    # Check if the user provided a model for etaH/etaV/zetaH/zetaV
    if isinstance(res, dict):
        res = res['res']

    # Check frequency
    freq = _check_var(freq, float, 1, 'freq')

    # Minimum frequency to avoid division by zero at freq = 0 Hz.
    # => min_freq can be set with utils.set_min
    freq = _check_min(freq, _min_freq, 'Frequencies', 'Hz', verb)
    if verb > 2:
        _prnt_min_max_val(freq, "   frequency  [Hz] : ", verb)

    # Calculate eta and zeta (horizontal and vertical)
    c = 299792458              # Speed of light m/s
    mu_0 = 4e-7*np.pi          # Magn. permeability of free space [H/m]
    epsilon_0 = 1./(mu_0*c*c)  # Elec. permittivity of free space [F/m]

    etaH = 1/res + np.outer(2j*np.pi*freq, epermH*epsilon_0)
    etaV = 1/(res*aniso*aniso) + np.outer(2j*np.pi*freq, epermV*epsilon_0)
    zetaH = np.outer(2j*np.pi*freq, mpermH*mu_0)
    zetaV = np.outer(2j*np.pi*freq, mpermV*mu_0)

    return freq, etaH, etaV, zetaH, zetaV


def check_hankel(ht, htarg, verb):
    r"""Check Hankel transform parameters.

    This check-function is called from one of the modelling routines in
    :mod:`model`.  Consult these modelling routines for a detailed description
    of the input parameters.

    Parameters
    ----------
    ht : {'fht', 'qwe', 'quad'}
        Flag to choose the Hankel transform.

    htarg : str or filter from empymod.filters or array_like,
        Depends on the value for ``ht``.

    verb : {0, 1, 2, 3, 4}
        Level of verbosity.


    Returns
    -------
    ht, htarg
        Checked if valid and set to defaults if not provided.

    """

    # Ensure ht is all lowercase
    ht = ht.lower()

    if ht == 'fht':    # If FHT, check filter settings

        # Get and check input or set defaults
        htarg = _check_targ(htarg, ['fhtfilt', 'pts_per_dec'])

        # Check filter; defaults to key_201_2009
        try:
            fhtfilt = htarg['fhtfilt']
            if not hasattr(fhtfilt, 'base'):
                fhtfilt = getattr(filters, fhtfilt)()
        except VariableCatch:
            fhtfilt = filters.key_201_2009()

        # Check pts_per_dec; defaults to 0
        try:
            pts_per_dec = _check_var(htarg['pts_per_dec'], float, 0,
                                     'fht: pts_per_dec', ())
        except VariableCatch:
            pts_per_dec = 0.0

        # Assemble htarg
        htarg = (fhtfilt, pts_per_dec)

        # If verbose, print Hankel transform information
        if verb > 2:
            print("   Hankel          :  DLF (Fast Hankel Transform)")
            print("     > Filter      :  " + fhtfilt.name)
            pstr = "     > DLF type    :  "
            if pts_per_dec < 0:
                print(pstr + "Lagged Convolution")
            elif pts_per_dec > 0:
                print(pstr + "Splined, " + str(pts_per_dec) + " pts/dec")
            else:
                print(pstr + "Standard")

    elif ht in ['qwe', 'hqwe']:
        # Rename ht
        ht = 'hqwe'

        # Get and check input or set defaults
        htarg = _check_targ(htarg, ['rtol', 'atol', 'nquad', 'maxint',
                            'pts_per_dec', 'diff_quad', 'a', 'b', 'limit'])

        # rtol : 1e-12
        try:
            rtol = _check_var(htarg['rtol'], float, 0, 'qwe: rtol', ())
        except VariableCatch:
            rtol = np.array(1e-12, dtype=float)

        # atol : 1e-30
        try:
            atol = _check_var(htarg['atol'], float, 0, 'qwe: atol', ())
        except VariableCatch:
            atol = np.array(1e-30, dtype=float)

        # nquad : 51
        try:
            nquad = _check_var(htarg['nquad'], int, 0, 'qwe: nquad', ())
        except VariableCatch:
            nquad = np.array(51, dtype=int)

        # maxint : 100
        try:
            maxint = _check_var(htarg['maxint'], int, 0, 'qwe: maxint', ())
        except VariableCatch:
            maxint = np.array(100, dtype=int)

        # pts_per_dec : 0  # No spline
        try:
            pts_per_dec = _check_var(htarg['pts_per_dec'], int, 0,
                                     'qwe: pts_per_dec', ())
            pts_per_dec = _check_min(pts_per_dec, 0, 'pts_per_dec', '', verb)
        except VariableCatch:
            pts_per_dec = np.array(0, dtype=int)

        # diff_quad : 100
        try:
            diff_quad = _check_var(htarg['diff_quad'], float, 0,
                                   'qwe: diff_quad', ())
        except VariableCatch:
            diff_quad = np.array(100, dtype=float)

        # a : None
        try:
            a = _check_var(htarg['a'], float, 0, 'qwe: a (quad)', ())
        except VariableCatch:
            a = None

        # b : None
        try:
            b = _check_var(htarg['b'], float, 0, 'qwe: b (quad)', ())
        except VariableCatch:
            b = None

        # limit : None
        try:
            limit = _check_var(htarg['limit'], float, 0, 'qwe: limit (quad)',
                               ())
        except VariableCatch:
            limit = None

        # Assemble htarg
        htarg = (rtol, atol, nquad, maxint, pts_per_dec, diff_quad, a, b,
                 limit)

        # If verbose, print Hankel transform information
        if verb > 2:
            print("   Hankel          :  Quadrature-with-Extrapolation")
            print("     > rtol        :  " + str(rtol))
            print("     > atol        :  " + str(atol))
            print("     > nquad       :  " + str(nquad))
            print("     > maxint      :  " + str(maxint))
            print("     > pts_per_dec :  " + str(pts_per_dec))
            print("     > diff_quad   :  " + str(diff_quad))
            if a:
                print("     > a     (quad):  " + str(a))
            if b:
                print("     > b     (quad):  " + str(b))
            if limit:
                print("     > limit (quad):  " + str(limit))

    elif ht in ['quad', 'hquad']:
        # Rename ht
        ht = 'hquad'

        # Get and check input or set defaults
        htarg = _check_targ(htarg, ['rtol', 'atol', 'limit', 'a', 'b',
                                    'pts_per_dec'])

        # rtol : 1e-12
        try:
            rtol = _check_var(htarg['rtol'], float, 0, 'quad: rtol', ())
        except VariableCatch:
            rtol = np.array(1e-12, dtype=float)

        # atol : 1e-20
        try:
            atol = _check_var(htarg['atol'], float, 0, 'quad: atol', ())
        except VariableCatch:
            atol = np.array(1e-20, dtype=float)

        # limit : 500
        try:
            limit = _check_var(htarg['limit'], int, 0, 'quad: limit', ())
        except VariableCatch:
            limit = np.array(500, dtype=int)

        # a : 1e-6
        try:
            a = _check_var(htarg['a'], float, 0, 'quad: a', ())
        except VariableCatch:
            a = np.array(1e-6, dtype=float)

        # b : 0.1
        try:
            b = _check_var(htarg['b'], float, 0, 'quad: b', ())
        except VariableCatch:
            b = np.array(0.1, dtype=float)

        # pts_per_dec : 40
        try:
            pts_per_dec = _check_var(htarg['pts_per_dec'], int, 0,
                                     'quad: pts_per_dec', ())
            pts_per_dec = _check_min(pts_per_dec, 1, 'pts_per_dec', '', verb)
        except VariableCatch:
            pts_per_dec = np.array(40, dtype=int)

        # Assemble htarg
        htarg = (rtol, atol, limit, a, b, pts_per_dec)

        # If verbose, print Hankel transform information
        if verb > 2:
            print("   Hankel          :  Quadrature")
            print("     > rtol        :  " + str(rtol))
            print("     > atol        :  " + str(atol))
            print("     > limit       :  " + str(limit))
            print("     > a           :  " + str(a))
            print("     > b           :  " + str(b))
            print("     > pts_per_dec :  " + str(pts_per_dec))

    else:
        print("* ERROR   :: <ht> must be one of: ['fht', 'qwe', 'quad'];" +
              " <ht> provided: " + str(ht))
        raise ValueError('ht')

    return ht, htarg


def check_model(depth, res, aniso, epermH, epermV, mpermH, mpermV, xdirect,
                verb):
    r"""Check the model: depth and corresponding layer parameters.

    This check-function is called from one of the modelling routines in
    :mod:`model`.  Consult these modelling routines for a detailed description
    of the input parameters.

    Parameters
    ----------
    depth : list
        Absolute layer interfaces z (m); #depth = #res - 1
        (excluding +/- infinity).

    res : array_like
        Horizontal resistivities rho_h (Ohm.m); #res = #depth + 1.

    aniso : array_like
        Anisotropies lambda = sqrt(rho_v/rho_h) (-); #aniso = #res.

    epermH, epermV : array_like
        Relative horizontal/vertical electric permittivities
        epsilon_h/epsilon_v (-);
        #epermH = #epermV = #res.

    mpermH, mpermV : array_like
        Relative horizontal/vertical magnetic permeabilities mu_h/mu_v (-);
        #mpermH = #mpermV = #res.

    xdirect : bool, optional
        If True and source and receiver are in the same layer, the direct field
        is calculated analytically in the frequency domain, if False it is
        calculated in the wavenumber domain.

    verb : {0, 1, 2, 3, 4}
        Level of verbosity.


    Returns
    -------
    depth : array
        Depths of layer interfaces, adds -infty at beginning if not present.

    res : array
        As input, checked for size.

    aniso : array
        As input, checked for size. If None, defaults to an array of ones.

    epermH, epermV : array_like
        As input, checked for size. If None, defaults to an array of ones.

    mpermH, mpermV : array_like
        As input, checked for size. If None, defaults to an array of ones.

    isfullspace : bool
        If True, the model is a fullspace (res, aniso, epermH, epermV, mpermM,
        and mpermV are in all layers the same).

    """
    global _min_res

    # Check depth
    if depth is None:
        depth = []
    depth = _check_var(depth, float, 1, 'depth')

    # Add -infinity at the beginning
    # => The top-layer (-infinity to first interface) is layer 0.
    if depth.size == 0:
        depth = np.array([-np.infty, ])
    elif depth[0] != -np.infty:
        depth = np.insert(depth, 0, -np.infty)

    # Ensure depth is increasing
    if np.any(depth[1:] - depth[:-1] < 0):
        print('* ERROR   :: <depth> must be increasing;' +
              ' <depth> provided: ' + _strvar(depth))
        raise ValueError('depth')

    # Check if the user provided a model for etaH/etaV/zetaH/zetaV
    if isinstance(res, dict):
        res_dict, res = res, res['res']
    else:
        res_dict = False

    # Cast and check resistivity
    res = _check_var(res, float, 1, 'res', depth.shape)
    # => min_res can be set with utils.set_min
    res = _check_min(res, _min_res, 'Resistivities', 'Ohm.m', verb)

    # Check optional parameters anisotropy, electric permittivity, and magnetic
    # permeability
    def check_inp(var, name, min_val):
        r"""Param-check function. Default to ones if not provided"""
        if var is None:
            return np.ones(depth.size)
        else:
            param = _check_var(var, float, 1, name, depth.shape)
            if name == 'aniso':  # Convert aniso into vertical resistivity
                param = param**2*res
            param = _check_min(param, min_val, 'Parameter ' + name, '', verb)
            if name == 'aniso':  # Convert vert. resistivity back to aniso
                param = np.sqrt(param/res)
            return param

    # => min_res can be set with utils.set_min
    aniso = check_inp(aniso, 'aniso', _min_res)
    epermH = check_inp(epermH, 'epermH', 0.0)
    epermV = check_inp(epermV, 'epermV', 0.0)
    mpermH = check_inp(mpermH, 'mpermH', 0.0)
    mpermV = check_inp(mpermV, 'mpermV', 0.0)

    # Print model parameters
    if verb > 2:
        print("   depth       [m] : ", _strvar(depth[1:]))
        print("   res     [Ohm.m] : ", _strvar(res))
        print("   aniso       [-] : ", _strvar(aniso))
        print("   epermH      [-] : ", _strvar(epermH))
        print("   epermV      [-] : ", _strvar(epermV))
        print("   mpermH      [-] : ", _strvar(mpermH))
        print("   mpermV      [-] : ", _strvar(mpermV))

    # Check if medium is a homogeneous full-space. If that is the case, the
    # EM-field is computed analytically directly in the frequency-domain.
    # Note: Also a stack of layers with the same material parameters is treated
    #       as a homogeneous full-space.
    isores = (res - res[0] == 0).all()*(aniso - aniso[0] == 0).all()
    isoep = (epermH - epermH[0] == 0).all()*(epermV - epermV[0] == 0).all()
    isomp = (mpermH - mpermH[0] == 0).all()*(mpermV - mpermV[0] == 0).all()
    isfullspace = isores*isoep*isomp

    # Check parameters of user-provided parameters
    if res_dict:
        # Switch off fullspace-option
        isfullspace = False

        # Loop over key, value pair and check
        for key, value in res_dict.items():
            if key not in ['res', 'func_eta', 'func_zeta']:
                res_dict[key] = check_inp(value, key, None)

        # Put res back
        res_dict['res'] = res

        # store res_dict back to res
        res = res_dict

    # Print fullspace info
    if verb > 2 and isfullspace:
        if xdirect:
            print("\n>  MODEL IS A FULLSPACE; returning analytical " +
                  "frequency-domain solution")
        else:
            print("\n>  MODEL IS A FULLSPACE")

    # Print xdirect info
    if verb > 2:
        if xdirect is None:
            print("   direct field    :  Not calculated (secondary field)")
        elif xdirect:
            print("   direct field    :  Calc. in frequency domain")
        else:
            print("   direct field    :  Calc. in wavenumber domain")

    return depth, res, aniso, epermH, epermV, mpermH, mpermV, isfullspace


def check_opt(opt, loop, ht, htarg, verb):
    r"""Check optimization parameters.

    This check-function is called from one of the modelling routines in
    :mod:`model`.  Consult these modelling routines for a detailed description
    of the input parameters.

    Parameters
    ----------
    opt : {None, 'parallel'}
        Optimization flag; use ``numexpr`` or not.

    loop : {None, 'freq', 'off'}
        Loop flag.

    ht : str
        Flag to choose the Hankel transform.

    htarg : array_like,
        Depends on the value for ``ht``.

    verb : {0, 1, 2, 3, 4}
        Level of verbosity.


    Returns
    -------
    use_ne_eval : bool
        Boolean if to use ``numexpr``.

    loop_freq : bool
        Boolean if to loop over frequencies.

    loop_off : bool
        Boolean if to loop over offsets.

    """

    # Check optimization flag
    use_ne_eval = False
    if opt == 'parallel':
        if numexpr:
            use_ne_eval = numexpr.evaluate
        elif verb > 0:
            print(numexpr_msg)

    # Define if to loop over frequencies or over offsets
    lagged_splined_fht = False
    if ht == 'fht':
        if htarg[1] != 0:
            lagged_splined_fht = True

    if ht in ['hqwe', 'hquad'] or lagged_splined_fht:
        loop_freq = True
        loop_off = False
    else:
        loop_off = loop == 'off'
        loop_freq = loop == 'freq'

    # If verbose, print optimization information
    if verb > 2:
        if use_ne_eval:
            print("   Kernel Opt.     :  Use parallel")
        else:
            print("   Kernel Opt.     :  None")

        if loop_off:
            print("   Loop over       :  Offsets")
        elif loop_freq:
            print("   Loop over       :  Frequencies")
        else:
            print("   Loop over       :  None (all vectorized)")

    return use_ne_eval, loop_freq, loop_off


def check_time(time, signal, ft, ftarg, verb):
    r"""Check time domain specific input parameters.

    This check-function is called from one of the modelling routines in
    :mod:`model`.  Consult these modelling routines for a detailed description
    of the input parameters.

    Parameters
    ----------
    time : array_like
        Times t (s).

    signal : {None, 0, 1, -1}
        Source signal:
            - None: Frequency-domain response
            - -1 : Switch-off time-domain response
            - 0 : Impulse time-domain response
            - +1 : Switch-on time-domain response

    ft : {'sin', 'cos', 'qwe', 'fftlog', 'fft'}
        Flag for Fourier transform.

    ftarg : str or filter from empymod.filters or array_like,
        Only used if ``signal`` !=None. Depends on the value for ``ft``:

    verb : {0, 1, 2, 3, 4}
        Level of verbosity.


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

    # Check time and input signal
    time = check_time_only(time, signal, verb)

    # Ensure ft is all lowercase
    ft = ft.lower()

    if ft in ['cos', 'sin', 'ffht']:  # Fourier-FHT (Sine/Cosine-filters)

        # If `ft='ffht'`, we assume that it run the check before, and get
        # sin/cos from ftarg. If not, defaults to 'sin'. To ensure that this
        # check can be re-run without failing.
        if ft == 'ffht':
            try:
                ft = ftarg[2]
            except VariableCatch:
                ft = 'sin'

        # If switch-off/on is required, ensure ft is cosine/sine
        if signal > 0:
            ft = 'sin'
        elif signal < 0:
            ft = 'cos'

        # Check Input
        ftarg = _check_targ(ftarg, ['fftfilt', 'pts_per_dec', 'ft'])

        # Check filter; defaults to key_201_CosSin_2012
        try:
            fftfilt = ftarg['fftfilt']
            if not hasattr(fftfilt, 'base'):
                fftfilt = getattr(filters, fftfilt)()
        except VariableCatch:
            fftfilt = filters.key_201_CosSin_2012()

        # Check pts_per_dec; defaults to -1
        try:
            pts_per_dec = _check_var(ftarg['pts_per_dec'], float, 0,
                                     ft + 'pts_per_dec', ())
        except VariableCatch:
            pts_per_dec = -1.0

        # Assemble ftarg
        ftarg = (fftfilt, pts_per_dec, ft)

        # If verbose, print Fourier transform information
        if verb > 2:
            if ft == 'sin':
                print("   Fourier         :  DLF (Sine-Filter)")
            else:
                print("   Fourier         :  DLF (Cosine-Filter)")
            print("     > Filter      :  " + fftfilt.name)
            pstr = "     > DLF type    :  "
            if pts_per_dec < 0:
                print(pstr + "Lagged Convolution")
            elif pts_per_dec > 0:
                print(pstr + "Splined, " + str(pts_per_dec) + " pts/dec")
            else:
                print(pstr + "Standard")

        # Get required frequencies
        # (multiply time by 2Pi, as calculation is done in angular frequencies)
        freq, _ = transform.get_spline_values(ftarg[0], 2*np.pi*time, ftarg[1])
        freq = np.squeeze(freq)

        # Rename ft
        ft = 'ffht'

    elif ft in ['qwe', 'fqwe']:       # QWE (using sine and imag-part)
        # Rename ft
        ft = 'fqwe'

        # Get and check input or set defaults
        ftarg = _check_targ(ftarg, ['rtol', 'atol', 'nquad', 'maxint',
                                    'pts_per_dec', 'diff_quad', 'a', 'b',
                                    'limit', 'sincos'])

        # If switch-off is required, use cosine, else sine
        if signal >= 0:
            sincos = np.sin
        elif signal < 0:
            sincos = np.cos

        try:  # rtol
            rtol = _check_var(ftarg['rtol'], float, 0, 'qwe: rtol', ())
        except VariableCatch:
            rtol = np.array(1e-8, dtype=float)

        try:  # atol
            atol = _check_var(ftarg['atol'], float, 0, 'qwe: atol', ())
        except VariableCatch:
            atol = np.array(1e-20, dtype=float)

        try:  # nquad
            nquad = _check_var(ftarg['nquad'], int, 0, 'qwe: nquad', ())
        except VariableCatch:
            nquad = np.array(21, dtype=int)

        try:  # maxint
            maxint = _check_var(ftarg['maxint'], int, 0, 'qwe: maxint', ())
        except VariableCatch:
            maxint = np.array(200, dtype=int)

        try:  # pts_per_dec
            pts_per_dec = _check_var(ftarg['pts_per_dec'], int, 0,
                                     'qwe: pts_per_dec', ())
            pts_per_dec = _check_min(pts_per_dec, 1, 'pts_per_dec', '', verb)
        except VariableCatch:
            pts_per_dec = np.array(20, dtype=int)

        # diff_quad : 100
        try:
            diff_quad = _check_var(ftarg['diff_quad'], int, 0,
                                   'qwe: diff_quad', ())
        except VariableCatch:
            diff_quad = np.array(100, dtype=int)

        # a : None
        try:
            a = _check_var(ftarg['a'], float, 0, 'qwe: a (quad)', ())
        except VariableCatch:
            a = None

        # b : None
        try:
            b = _check_var(ftarg['b'], float, 0, 'qwe: b (quad)', ())
        except VariableCatch:
            b = None

        # limit : None
        try:
            limit = _check_var(ftarg['limit'], float, 0, 'qwe: limit (quad)',
                               ())
        except VariableCatch:
            limit = None

        # Assemble ftarg
        ftarg = (rtol, atol, nquad, maxint, pts_per_dec, diff_quad, a, b,
                 limit, sincos)

        # If verbose, print Fourier transform information
        if verb > 2:
            print("   Fourier         :  Quadrature-with-Extrapolation")
            print("     > rtol        :  " + str(rtol))
            print("     > atol        :  " + str(atol))
            print("     > nquad       :  " + str(nquad))
            print("     > maxint      :  " + str(maxint))
            print("     > pts_per_dec :  " + str(pts_per_dec))
            print("     > diff_quad   :  " + str(diff_quad))
            if a:
                print("     > a     (quad):  " + str(a))
            if b:
                print("     > b     (quad):  " + str(b))
            if limit:
                print("     > limit (quad):  " + str(limit))

        # Get required frequencies
        g_x, _ = special.p_roots(nquad)
        minf = np.floor(10*np.log10((g_x.min() + 1)*np.pi/2/time.max()))/10
        maxf = np.ceil(10*np.log10(maxint*np.pi/time.min()))/10
        freq = np.logspace(minf, maxf, (maxf-minf)*pts_per_dec + 1)

    elif ft == 'fftlog':              # FFTLog (using sine and imag-part)

        # Get and check input or set defaults
        ftarg = _check_targ(ftarg, ['pts_per_dec', 'add_dec', 'q', 'mu',
                                    'tcalc', 'dlnr', 'kr', 'rk'])

        try:  # pts_per_dec
            pts_per_dec = _check_var(ftarg['pts_per_dec'], int, 0,
                                     'fftlog: pts_per_dec', ())
            pts_per_dec = _check_min(pts_per_dec, 1, 'pts_per_dec', '', verb)
        except VariableCatch:
            pts_per_dec = np.array(10, dtype=int)

        try:  # add_dec
            add_dec = _check_var(ftarg['add_dec'], float, 1, 'fftlog: add_dec',
                                 (2,))
        except VariableCatch:
            add_dec = np.array([-2, 1], dtype=float)

        try:  # q
            q = _check_var(ftarg['q'], float, 0, 'fftlog: q', ())
            # Restrict q to +/- 1
            if np.abs(q) > 1:
                q = np.sign(q)
        except VariableCatch:
            q = np.array(0, dtype=float)

        # If switch-off is required, use cosine, else sine
        if signal >= 0:
            mu = 0.5
        elif signal < 0:
            mu = -0.5

        # If verbose, print Fourier transform information
        if verb > 2:
            print("   Fourier         :  FFTLog")
            print("     > pts_per_dec :  " + str(pts_per_dec))
            print("     > add_dec     :  " + str(add_dec))
            print("     > q           :  " + str(q))

        # Calculate minimum and maximum required frequency
        minf = np.log10(1/time.max()) + add_dec[0]
        maxf = np.log10(1/time.min()) + add_dec[1]
        n = np.int(maxf - minf)*pts_per_dec

        # Initialize FFTLog, get required parameters
        freq, tcalc, dlnr, kr, rk = transform.fhti(minf, maxf, n, q, mu)

        # Assemble ftarg
        # Keep first 3 entries, so re-running this check is stable
        ftarg = (pts_per_dec, add_dec, q, mu, tcalc, dlnr, kr, rk)

    elif ft == 'fft':                 # FFT

        # Get and check input or set defaults
        ftarg = _check_targ(ftarg, ['dfreq', 'nfreq', 'ntot', 'pts_per_dec',
                                    'fftfreq'])

        try:  # dfreq
            dfreq = _check_var(ftarg['dfreq'], float, 0, 'fft: dfreq', ())
        except VariableCatch:
            dfreq = np.array(0.002, dtype=float)

        try:  # nfreq
            nfreq = _check_var(ftarg['nfreq'], int, 0, 'fft: nfreq', ())
        except VariableCatch:
            nfreq = np.array(2048, dtype=int)

        nall = 2**np.arange(30)
        try:  # ntot
            ntot = _check_var(ftarg['ntot'], int, 0, 'fft: ntot', ())
        except VariableCatch:
            # We could use here fftpack.next_fast_len, but tests have shown
            # that powers of two yield better results in this case.
            ntot = nall[np.argmax(nall >= nfreq)]
        else:  # Assure that input ntot is not bigger than nfreq
            if nfreq > ntot:
                ntot = nall[np.argmax(nall >= nfreq)]

        # Check pts_per_dec; defaults to None
        try:
            pts_per_dec = _check_var(ftarg['pts_per_dec'], int, 0,
                                     'fft: pts_per_dec', ())
            pts_per_dec = _check_min(pts_per_dec, 1, 'pts_per_dec', '', verb)
        except VariableCatch:
            pts_per_dec = None

        # Get required frequencies
        if pts_per_dec:  # Space actually calculated freqs logarithmically.
            start = np.log10(dfreq)
            stop = np.log10(nfreq*dfreq)
            freq = np.logspace(start, stop, (stop-start)*pts_per_dec + 1)
        else:
            freq = np.arange(1, nfreq+1)*dfreq

        # Assemble ftarg
        ftarg = (dfreq, nfreq, ntot, pts_per_dec)

        # If verbose, print Fourier transform information
        if verb > 2:
            print("   Fourier         :  Fast Fourier Transform FFT")
            print("     > dfreq       :  " + str(ftarg[0]))
            print("     > nfreq       :  " + str(ftarg[1]))
            print("     > ntot        :  " + str(ftarg[2]))
            if pts_per_dec:
                print("     > pts_per_dec :  " + str(ftarg[3]))
            else:
                print("     > pts_per_dec :  (linear)")

    else:
        print("* ERROR   :: <ft> must be one of: ['cos', 'sin', 'qwe', " +
              "'fftlog', 'fft']; <ft> provided: "+str(ft))
        raise ValueError('ft')

    return time, freq, ft, ftarg


def check_time_only(time, signal, verb):
    r"""Check time and signal parameters.

    This check-function is called from one of the modelling routines in
    :mod:`model`.  Consult these modelling routines for a detailed description
    of the input parameters.

    Parameters
    ----------
    time : array_like
        Times t (s).

    signal : {None, 0, 1, -1}
        Source signal:
            - None: Frequency-domain response
            - -1 : Switch-off time-domain response
            - 0 : Impulse time-domain response
            - +1 : Switch-on time-domain response

    verb : {0, 1, 2, 3, 4}
        Level of verbosity.


    Returns
    -------
    time : float
        Time, checked for size and assured min_time.

    """
    global _min_time

    # Check input signal
    if int(signal) not in [-1, 0, 1]:
        print("* ERROR   :: <signal> must be one of: [None, -1, 0, 1]; " +
              "<signal> provided: "+str(signal))
        raise ValueError('signal')

    # Check time
    time = _check_var(time, float, 1, 'time')

    # Minimum time to avoid division by zero  at time = 0 s.
    # => min_time can be set with utils.set_min
    time = _check_min(time, _min_time, 'Times', 's', verb)
    if verb > 2:
        _prnt_min_max_val(time, "   time        [s] : ", verb)

    return time


def check_solution(solution, signal, ab, msrc, mrec):
    r"""Check required solution with parameters.

    This check-function is called from one of the modelling routines in
    :mod:`model`. Consult these modelling routines for a detailed description
    of the input parameters.

    Parameters
    ----------
    solution : str
        String to define analytical solution.

    signal : {None, 0, 1, -1}
        Source signal:
            - None: Frequency-domain response
            - -1 : Switch-off time-domain response
            - 0 : Impulse time-domain response
            - +1 : Switch-on time-domain response

    msrc, mrec : bool
        True if src/rec is magnetic, else False.

    """

    # Ensure valid solution.
    if solution not in ['fs', 'dfs', 'dhs', 'dsplit', 'dtetm']:
        print("* ERROR   :: Solution must be one of ['fs', 'dfs', 'dhs', " +
              "'dsplit', 'dtetm']; <solution> provided: " + solution)
        raise ValueError('solution')

    # If diffusive solution is required, ensure EE-field.
    if solution[0] == 'd' and (msrc or mrec):
        print('* ERROR   :: Diffusive solution is only implemented for ' +
              'electric sources and electric receivers, <ab> provided: ' +
              str(ab))
        raise ValueError('ab')

    # If full solution is required, ensure frequency-domain.
    if solution == 'fs' and signal is not None:
        print('* ERROR   :: Full fullspace solution is only implemented for ' +
              'the frequency domain, <signal> provided: ' + str(signal))
        raise ValueError('signal')


# 2.b <Get>s (alphabetically)

def get_abs(msrc, mrec, srcazm, srcdip, recazm, recdip, verb):
    r"""Get required ab's for given angles.

    This check-function is called from one of the modelling routines in
    :mod:`model`.  Consult these modelling routines for a detailed description
    of the input parameters.

    Parameters
    ----------
    msrc, mrec : bool
        True if src/rec is magnetic, else False.

    srcazm, recazm : float
        Horizontal source/receiver angle (azimuth).

    srcdip, recdip : float
        Vertical source/receiver angle (dip).

    verb : {0, 1, 2, 3, 4}
        Level of verbosity.


    Returns
    -------
    ab_calc : array of int
        ab's to calculate for this bipole.

    """

    # Get required ab's (9 at most)
    ab_calc = np.array([[11, 12, 13], [21, 22, 23], [31, 32, 33]])
    if msrc:
        ab_calc += 3
    if mrec:
        ab_calc += 30

        # Switch <ab> using reciprocity.
        if msrc:
            # G^mm_ab(s, r, e, z) = -G^ee_ab(s, r, -z, -e)
            ab_calc -= 33  # -30 : mrec->erec; -3: msrc->esrc
        else:
            # G^me_ab(s, r, e, z) = -G^em_ba(r, s, e, z)
            ab_calc = ab_calc % 10*10 + ab_calc // 10  # Swap alpha/beta

    # Remove unnecessary ab's
    bab = np.asarray(ab_calc*0+1, dtype=bool)

    # Remove if source is x- or y-directed
    check = np.atleast_1d(srcazm)[0]
    if np.allclose(srcazm % (np.pi/2), 0):  # if all angles are multiples of 90
        if np.isclose(check // (np.pi/2) % 2, 0):  # Multiples of pi (180)
            bab[:, 1] *= False        # x-directed source, remove y
        else:                                      # Multiples of pi/2 (90)
            bab[:, 0] *= False        # y-directed source, remove x

    # Remove if source is vertical
    check = np.atleast_1d(srcdip)[0]
    if np.allclose(srcdip % (np.pi/2), 0):  # if all angles are multiples of 90
        if np.isclose(check // (np.pi/2) % 2, 0):  # Multiples of pi (180)
            bab[:, 2] *= False        # Horizontal, remove z
        else:                                      # Multiples of pi/2 (90)
            bab[:, :2] *= False       # Vertical, remove x/y

    # Remove if receiver is x- or y-directed
    check = np.atleast_1d(recazm)[0]
    if np.allclose(recazm % (np.pi/2), 0):  # if all angles are multiples of 90
        if np.isclose(check // (np.pi/2) % 2, 0):  # Multiples of pi (180)
            bab[1, :] *= False        # x-directed receiver, remove y
        else:                                      # Multiples of pi/2 (90)
            bab[0, :] *= False        # y-directed receiver, remove x

    # Remove if receiver is vertical
    check = np.atleast_1d(recdip)[0]
    if np.allclose(recdip % (np.pi/2), 0):  # if all angles are multiples of 90
        if np.isclose(check // (np.pi/2) % 2, 0):  # Multiples of pi (180)
            bab[2, :] *= False        # Horizontal, remove z
        else:                                      # Multiples of pi/2 (90)
            bab[:2, :] *= False       # Vertical, remove x/y

    # Reduce
    ab_calc = ab_calc[bab].ravel()

    # Print actual calculated <ab>
    if verb > 2:
        print("   Required ab's   : ", _strvar(ab_calc))

    return ab_calc


def get_geo_fact(ab, srcazm, srcdip, recazm, recdip, msrc, mrec):
    r"""Get required geometrical scaling factor for given angles.

    This check-function is called from one of the modelling routines in
    :mod:`model`.  Consult these modelling routines for a detailed description
    of the input parameters.

    Parameters
    ----------
    ab : int
        Source-receiver configuration.

    srcazm, recazm : float
        Horizontal source/receiver angle.

    srcdip, recdip : float
        Vertical source/receiver angle.


    Returns
    -------
    fact : float
        Geometrical scaling factor.

    """
    global _min_angle

    # Get current direction for source and receiver
    fis = ab % 10
    fir = ab // 10

    # If rec is magnetic and src not, swap directions (reciprocity).
    # (They have been swapped in get_abs, but the original scaling applies.)
    if mrec and not msrc:
        fis, fir = fir, fis

    def gfact(bp, azm, dip):
        r"""Geometrical factor of source or receiver."""
        if bp in [1, 4]:    # x-directed
            return np.cos(azm)*np.cos(dip)
        elif bp in [2, 5]:  # y-directed
            return np.sin(azm)*np.cos(dip)
        else:               # z-directed
            return np.sin(dip)

    # Calculate src-rec-factor
    fsrc = gfact(fis, srcazm, srcdip)
    frec = gfact(fir, recazm, recdip)
    fact = np.outer(fsrc, frec).ravel()

    # Set very small angles to proper zero (because e.g. sin(pi/2) != exact 0)
    # => min_angle can be set with utils.set_min
    fact[np.abs(fact) < _min_angle] = 0

    return fact


def get_layer_nr(inp, depth):
    r"""Get number of layer in which inp resides.

    Note:
    If zinp is on a layer interface, the layer above the interface is chosen.

    This check-function is called from one of the modelling routines in
    :mod:`model`.  Consult these modelling routines for a detailed description
    of the input parameters.

    Parameters
    ----------
    inp : list of floats or arrays
        Dipole coordinates (m)

    depth : array
        Depths of layer interfaces.


    Returns
    -------
    linp : int or array_like of int
        Layer number(s) in which inp resides (plural only if bipole).

    zinp : float or array
        inp[2] (depths).

    """
    zinp = inp[2]

    #  depth = [-infty : last interface]; create additional depth-array
    # pdepth = [fist interface : +infty]
    pdepth = np.concatenate((depth[1:], np.array([np.infty])))

    # Broadcast arrays
    b_zinp = np.atleast_1d(zinp)[:, None]

    # Get layers
    linp = np.where((depth[None, :] < b_zinp)*(pdepth[None, :] >= b_zinp))[1]

    # Return; squeeze in case of only one inp-depth
    return np.squeeze(linp), zinp


def get_off_ang(src, rec, nsrc, nrec, verb):
    r"""Get depths, offsets, angles, hence spatial input parameters.

    This check-function is called from one of the modelling routines in
    :mod:`model`.  Consult these modelling routines for a detailed description
    of the input parameters.

    Parameters
    ----------
    src, rec : list of floats or arrays
        Source/receiver dipole coordinates x, y, and z (m).

    nsrc, nrec : int
        Number of sources/receivers (-).

    verb : {0, 1, 2, 3, 4}
        Level of verbosity.


    Returns
    -------
    off : array of floats
        Offsets

    angle : array of floats
        Angles

    """
    global _min_off

    # Pre-allocate off and angle
    off = np.empty((nrec*nsrc,))
    angle = np.empty((nrec*nsrc,))

    # Coordinates
    # Loop over sources, append them one after another.
    for i in range(nsrc):
        xco = rec[0] - src[0][i]  # X-coordinates  [m]
        yco = rec[1] - src[1][i]  # Y-coordinates  [m]
        off[i*nrec:(i+1)*nrec] = np.sqrt(xco*xco + yco*yco)  # Offset   [m]
        angle[i*nrec:(i+1)*nrec] = np.arctan2(yco, xco)      # Angle  [rad]

    # Note: One could achieve a potential speed-up using np.unique to sort out
    # src-rec configurations that have the same offset and angle. Very unlikely
    # for real data.

    # Minimum offset to avoid singularities at off = 0 m.
    # => min_off can be set with utils.set_min
    angle[np.where(off < _min_off)] = np.nan
    off = _check_min(off, _min_off, 'Offsets', 'm', verb)

    return off, angle


def get_azm_dip(inp, iz, ninpz, intpts, isdipole, strength, name, verb):
    r"""Get angles, interpolation weights and normalization weights.

    This check-function is called from one of the modelling routines in
    :mod:`model`.  Consult these modelling routines for a detailed description
    of the input parameters.

    Parameters
    ----------
    inp : list of floats or arrays
        Input coordinates (m):
            - [x0, x1, y0, y1, z0, z1] (bipole of finite length)
            - [x, y, z, azimuth, dip]  (dipole, infinitesimal small)

    iz : int
        Index of current di-/bipole depth (-).

    ninpz : int
        Total number of di-/bipole depths (ninpz = 1 or npinz = nsrc) (-).

    intpts : int
        Number of integration points for bipole (-).

    isdipole : bool
        Boolean if inp is a dipole.

    strength : float, optional
        Source strength (A):
          - If 0, output is normalized to source and receiver of 1 m length,
            and source strength of 1 A.
          - If != 0, output is returned for given source and receiver length,
            and source strength.

    name : str, {'src', 'rec'}
        Pole-type.

    verb : {0, 1, 2, 3, 4}
        Level of verbosity.


    Returns
    -------
    tout : list of floats or arrays
        Dipole coordinates x, y, and z (m).

    azm : float or array of floats
        Horizontal angle (azimuth).

    dip : float or array of floats
        Vertical angle (dip).

    g_w : float or array of floats
        Factors from Gaussian interpolation.

    intpts : int
        As input, checked.

    inp_w : float or array of floats
        Factors from source/receiver length and source strength.

    """
    global _min_off

    # Get this di-/bipole
    if ninpz == 1:  # If there is only one distinct depth, all at once
        tinp = inp
    else:  # If there are several depths, we take the current one
        if isdipole:
            tinp = [np.atleast_1d(inp[0][iz]), np.atleast_1d(inp[1][iz]),
                    np.atleast_1d(inp[2][iz]), np.atleast_1d(inp[3]),
                    np.atleast_1d(inp[4])]
        else:
            tinp = [inp[0][iz], inp[1][iz], inp[2][iz],
                    inp[3][iz], inp[4][iz], inp[5][iz]]

    # Check source strength variable
    strength = _check_var(strength, float, 0, 'strength', ())

    # Dipole/Bipole specific
    if isdipole:

        # If input is a dipole, set intpts to 1
        intpts = 1

        # Check azm
        azm = _check_var(np.deg2rad(tinp[3]), float, 1, 'azimuth')

        # Check dip
        dip = _check_var(np.deg2rad(tinp[4]), float, 1, 'dip')

        # If dipole, g_w are ones
        g_w = np.ones(tinp[0].size)

        # If dipole, inp_w are once, unless strength > 0
        inp_w = np.ones(tinp[0].size)
        if name == 'src' and strength > 0:
            inp_w *= strength

        # Collect output
        tout = tinp

    else:
        # Get lengths in each direction
        dx = np.squeeze(tinp[1] - tinp[0])
        dy = np.squeeze(tinp[3] - tinp[2])
        dz = np.squeeze(tinp[5] - tinp[4])

        # Length of bipole
        dl = np.atleast_1d(np.linalg.norm([dx, dy, dz], axis=0))

        # Horizontal deviation from x-axis
        azm = np.atleast_1d(np.arctan2(dy, dx))

        # Vertical deviation from xy-plane down
        dip = np.atleast_1d(np.pi/2-np.arccos(dz/dl))

        # Check intpts
        intpts = _check_var(intpts, int, 0, 'intpts', ())

        # Gauss quadrature if intpts > 2; else set to center of tinp
        if intpts > 2:  # Calculate the dipole positions
            # Get integration positions and weights
            g_x, g_w = special.p_roots(intpts)
            g_x = np.outer(g_x, dl/2.0)  # Adjust to tinp length
            g_w /= 2.0  # Adjust to tinp length (dl/2), normalize (1/dl)

            # Coordinate system is left-handed, positive z down
            # (East-North-Depth).
            xinp = tinp[0] + dx/2 + g_x*np.cos(dip)*np.cos(azm)
            yinp = tinp[2] + dy/2 + g_x*np.cos(dip)*np.sin(azm)
            zinp = tinp[4] + dz/2 + g_x*np.sin(dip)

            # Reduce zinp to one, if ninpz is 1 (as they are all the same then)
            if ninpz == 1:
                zinp = zinp[:, 0]

        else:  # If intpts < 3: Calculate bipole at tinp-centre for dip/azm

            # Set intpts to 1
            intpts = 1

            # Get centre points
            xinp = np.array(tinp[0] + dx/2)
            yinp = np.array(tinp[2] + dy/2)
            zinp = np.array(tinp[4] + dz/2)

            # Gaussian weights in this case are ones
            g_w = np.array([1])

        # Scaling
        inp_w = np.ones(dl.size)
        if strength > 0:  # If strength > 0, we scale it by bipole-length
            inp_w *= dl
            if name == 'src':  # If source, additionally by source strength
                inp_w *= strength

        # Collect output list; rounding coord. to same precision as min_off
        rndco = int(np.round(np.log10(1/_min_off)))
        tout = [np.round(xinp, rndco).ravel('F'),
                np.round(yinp, rndco).ravel('F'),
                np.round(zinp, rndco).ravel('F')]

    # Print spatial parameters
    if verb > 2:
        # Pole-type: src or rec
        if name == 'src':
            longname = '   Source(s)       : '
        else:
            longname = '   Receiver(s)     : '

        # Print dipole/bipole information
        if isdipole:
            print(longname, str(tout[0].size), 'dipole(s)')
            tname = ['x  ', 'y  ', 'z  ']
            prntinp = tout
        else:
            print(longname, str(int(tout[0].size/intpts)), 'bipole(s)')
            tname = ['x_c', 'y_c', 'z_c']
            if intpts == 1:
                print("     > intpts      :  1 (as dipole)")
                prntinp = tout
            else:
                print("     > intpts      : ", intpts)
                prntinp = [np.atleast_1d(tinp[0])[0] + dx/2,
                           np.atleast_1d(tinp[2])[0] + dy/2,
                           np.atleast_1d(tinp[4])[0] + dz/2]

            # Print bipole length and strength
            _prnt_min_max_val(dl, "     > length  [m] : ", verb)
            print("     > strength[A] : ", _strvar(strength))

        # Print coordinates
        for i in range(3):
            text = "     > " + tname[i] + "     [m] : "
            _prnt_min_max_val(prntinp[i], text, verb)

        # Print angles
        _prnt_min_max_val(np.rad2deg(azm), "     > azimuth [°] : ", verb)
        _prnt_min_max_val(np.rad2deg(dip), "     > dip     [°] : ", verb)

    return tout, azm, dip, g_w, intpts, inp_w


def printstartfinish(verb, inp=None, kcount=None):
    r"""Print start and finish with time measure and kernel count."""
    if inp:
        if verb > 1:
            ttxt = str(timedelta(seconds=default_timer() - inp))
            ktxt = ' '
            if kcount:
                ktxt += str(kcount) + ' kernel call(s)'
            print('\n:: empymod END; runtime = ' + ttxt + ' ::' + ktxt + '\n')
    else:
        t0 = default_timer()
        if verb > 2:
            print("\n:: empymod START  ::\n")
        return t0


def conv_warning(conv, targ, name, verb):
    r"""Print error if QWE/QUAD did not converge at least once."""
    if verb > 0 and not conv:
        print('* WARNING :: ' + name +
              '-quadrature did not converge at least once;\n             ' +
              '=> desired `atol` and `rtol` might not be achieved.')


# 3. Set/get min values

def set_minimum(min_freq=None, min_time=None, min_off=None, min_res=None,
                min_angle=None):
    r"""
    Set minimum values of parameters.

    The given parameters are set to its minimum value if they are smaller.

    Parameters
    ----------
    min_freq : float, optional
        Minimum frequency [Hz] (default 1e-20 Hz).
    min_time : float, optional
        Minimum time [s] (default 1e-20 s).
    min_off : float, optional
        Minimum offset [m] (default 1e-3 m).
        Also used to round src- & rec-coordinates.
    min_res : float, optional
        Minimum horizontal and vertical resistivity [Ohm.m] (default 1e-20).
    min_angle : float, optional
        Minimum angle [-] (default 1e-10).

    Note
    ----
    set_minimum and get_minimum are derived after set_printoptions and
    get_printoptions from arrayprint.py in numpy.

    """

    global _min_freq, _min_time, _min_off, _min_res, _min_angle

    if min_freq is not None:
        _min_freq = min_freq
    if min_time is not None:
        _min_time = min_time
    if min_off is not None:
        _min_off = min_off
    if min_res is not None:
        _min_res = min_res
    if min_angle is not None:
        _min_angle = min_angle


def get_minimum():
    r"""
    Return the current minimum values.

    Returns
    -------
    min_vals : dict
        Dictionary of current minimum values with keys

          - min_freq : float
          - min_time : float
          - min_off : float
          - min_res : float
          - min_angle : float

        For a full description of these options, see `set_minimum`.

    Note
    ----
    set_minimum and get_minimum are derived after set_printoptions and
    get_printoptions from arrayprint.py in numpy.

    """
    d = dict(min_freq=_min_freq,
             min_time=_min_time,
             min_off=_min_off,
             min_res=_min_res,
             min_angle=_min_angle)
    return d


# 4. Internal utilities

def _check_shape(var, name, shape, shape2=None):
    r"""Check that <var> has shape <shape>; if false raise ValueError(name)"""
    varshape = np.shape(var)
    if shape != varshape:
        if shape2:
            if shape2 != varshape:
                print('* ERROR   :: Parameter ' + name + ' has wrong shape!' +
                      ' : ' + str(varshape) + ' instead of ' + str(shape) +
                      'or' + str(shape2) + '.')
                raise ValueError(name)
        else:
            print('* ERROR   :: Parameter ' + name + ' has wrong shape! : ' +
                  str(varshape) + ' instead of ' + str(shape) + '.')
            raise ValueError(name)


def _check_var(var, dtype, ndmin, name, shape=None, shape2=None):
    r"""Return variable as array of dtype, ndmin; shape-checked."""
    if var is None:
        raise ValueError
    var = np.array(var, dtype=dtype, copy=True, ndmin=ndmin)
    if shape:
        _check_shape(var, name, shape, shape2)
    return var


def _strvar(a, prec='{:G}'):
    r"""Return variable as a string to print, with given precision."""
    return ' '.join([prec.format(i) for i in np.atleast_1d(a)])


def _prnt_min_max_val(var, text, verb):
    r"""Print variable; if more than three, just min/max, unless verb > 3."""
    if var.size > 3:
        print(text, _strvar(var.min()), "-", _strvar(var.max()),
              ":", _strvar(var.size), " [min-max; #]")
        if verb > 3:
            print("                   : ", _strvar(var))
    else:
        print(text, _strvar(np.atleast_1d(var)))


def _check_min(par, minval, name, unit, verb):
    r"""Check minimum value of parameter."""
    scalar = False
    if par.shape == ():
        scalar = True
        par = np.atleast_1d(par)
    if minval is not None:
        ipar = np.where(par < minval)
        par[ipar] = minval
        if verb > 0 and np.size(ipar) != 0:
            print('* WARNING :: ' + name + ' < ' + str(minval) + ' ' + unit +
                  ' are set to ' + str(minval) + ' ' + unit + '!')
    if scalar:
        return np.squeeze(par)
    else:
        return par


def _check_targ(targ, keys):
    r"""Check format of htarg/ftarg and return dict."""
    if not targ:  # If None
        targ = {}
    elif not isinstance(targ, (list, tuple, dict)):  # If only one value
        targ = [targ, ]
    if isinstance(targ, (list, tuple)):  # Put list into dict
        targ = {keys[i]: targ[i] for i in range(min(len(targ), len(keys)))}
    return targ


# 5. Backwards compatibility

def spline_backwards_hankel(ht, htarg, opt):
    r"""Check opt if deprecated 'spline' is used.

    Returns corrected htarg, opt.
    r"""
    # Ensure ht is all lowercase
    ht = ht.lower()

    # Only relevant for 'fht' and 'hqwe', not for 'quad'
    if ht in ['fht', 'qwe', 'hqwe']:

        # Get corresponding htarg
        if ht == 'fht':
            htarg = _check_targ(htarg, ['fhtfilt', 'pts_per_dec'])
        elif ht in ['qwe', 'hqwe']:
            htarg = _check_targ(htarg, ['rtol', 'atol', 'nquad', 'maxint',
                                'pts_per_dec', 'diff_quad', 'a', 'b', 'limit'])

        # If spline (qwe, fht) or lagged (fht)
        if opt == 'spline':

            # Issue warning
            mesg = ("\n    The use of `opt='spline'` is deprecated and will " +
                    "be removed\n    in v2.0.0; use the corresponding " +
                    "setting in `htarg`.")
            warnings.warn(mesg, DeprecationWarning)

            # Reset opt
            opt = None

            # Check pts_per_dec; set to old default values if not given
            if 'pts_per_dec' not in htarg:
                if ht == 'fht':
                    htarg['pts_per_dec'] = -1  # Lagged Convolution DLF

                elif ht in ['qwe', 'hqwe']:
                    htarg['pts_per_dec'] = 80  # Splined QWE; old default value

    return htarg, opt
