"""
Utilities for :mod:`empymod.model` such as checking input parameters.

This module consists of four groups of functions:
   0. General settings
   1. Class EMArray
   2. Input parameter checks for modelling
   3. Internal utilities

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


# Mandatory imports
import copy
import numpy as np
from scipy import special
from timeit import default_timer
from datetime import timedelta, datetime

# Relative imports
from empymod import filters, transform

# scooby is a soft dependency for empymod
try:
    from scooby import Report as ScoobyReport
except ImportError:
    class ScoobyReport:
        def __init__(self, additional, core, optional, ncol, text_width, sort):
            print("\n* WARNING :: `empymod.Report` requires `scooby`."
                  "\n             Install it via `pip install scooby`.\n")

# Version: We take care of it here instead of in __init__, so we can use it
# within the package itself (logs).
try:
    # - Released versions just tags:       1.10.0
    # - GitHub commits add .dev#+hash:     1.10.1.dev3+g973038c
    # - Uncommitted changes add timestamp: 1.10.1.dev3+g973038c.d20191022
    from empymod.version import version as __version__
except ImportError:
    # If it was not installed, then we don't know the version. We could throw a
    # warning here, but this case *should* be rare. empymod should be installed
    # properly!
    __version__ = 'unknown-'+datetime.today().strftime('%Y%m%d')


__all__ = ['EMArray', 'check_time_only', 'check_time', 'check_model',
           'check_frequency', 'check_hankel', 'check_loop', 'check_dipole',
           'check_bipole', 'check_ab', 'check_solution', 'get_abs',
           'get_geo_fact', 'get_azm_dip', 'get_off_ang', 'get_layer_nr',
           'printstartfinish', 'conv_warning', 'set_minimum', 'get_minimum',
           'Report']

# 0. General settings

_min_freq = 1e-20   # Minimum frequency  [Hz]
_min_time = 1e-20   # Minimum time       [s]
_min_off = 1e-3     # Minimum offset     [m]
#                   # > Also used to round src- & rec-coordinates (1e-3 => mm)
_min_res = 1e-20    # Minimum value for horizontal/vertical resistivity
_min_angle = 1e-10  # Angle factors smaller than that are set to 0


# 1. Class EMArray

class EMArray(np.ndarray):
    r"""Create an EM-ndarray: add *amplitude* <amp> and *phase* <pha> methods.

    Parameters
    ----------
    data : array
        Data to which to add `.amp` and `.pha` attributes.


    Examples
    --------
    >>> import numpy as np
    >>> from empymod.utils import EMArray
    >>> emvalues = EMArray(np.array([1+1j, 1-4j, -1+2j]))
    >>> print(f"Amplitude         : {emvalues.amp()}")
    Amplitude         : [1.41421356 4.12310563 2.23606798]
    >>> print(f"Phase (rad)       : {emvalues.pha()}")
    Phase (rad)       : [ 0.78539816 -1.32581766 -4.24874137]
    >>> print(f"Phase (deg)       : {emvalues.pha(deg=True)}")
    Phase (deg)       : [  45.          -75.96375653 -243.43494882]
    >>> print(f"Phase (deg; lead) : {emvalues.pha(deg=True, lag=False)}")
    Phase (deg; lead) : [-45.          75.96375653 243.43494882]

    """

    def __new__(cls, data):
        r"""Create a new EMArray."""
        return np.asarray(data).view(cls)

    def amp(self):
        """Amplitude of the electromagnetic field."""
        return np.abs(self.view())

    def pha(self, deg=False, unwrap=True, lag=True):
        """Phase of the electromagnetic field.

        Parameters
        ----------
        deg : bool
            If True the returned phase is in degrees, else in radians.
            Default is False (radians).

        unwrap : bool
            If True the returned phase is unwrapped.
            Default is True (unwrapped).

        lag : bool
            If True the returned phase is lag, else lead defined.
            Default is True (lag defined).

        """
        # Get phase, lead or lag defined.
        if lag:
            pha = np.angle(self.view())
        else:
            pha = np.angle(np.conj(self.view()))

        # Unwrap if `unwrap`.
        # np.unwrap removes the EMArray class;
        # for consistency, we wrap it in EMArray again.
        if unwrap and self.size > 1:
            pha = EMArray(np.unwrap(pha))

        # Convert to degrees if `deg`.
        if deg:
            pha *= 180/np.pi

        return pha


# 2. Input parameter checks for modelling

# 2.a <Check>s (alphabetically)

def check_ab(ab, verb):
    r"""Check source-receiver configuration.

    This check-function is called from one of the modelling routines in
    :mod:`empymod.model`. Consult these modelling routines for a detailed
    description of the input parameters.

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
    except TypeError as e:
        raise TypeError("<ab> must be an integer.") from e

    # Check src and rec orientation (<ab> for alpha-beta)
    # pab: all possible values that <ab> can take
    pab = [11, 12, 13, 14, 15, 16, 21, 22, 23, 24, 25, 26,
           31, 32, 33, 34, 35, 36, 41, 42, 43, 44, 45, 46,
           51, 52, 53, 54, 55, 56, 61, 62, 63, 64, 65, 66]
    if ab not in pab:
        raise ValueError(f"<ab> must be one of: {pab}; <ab> provided: {ab}.")

    # Print input <ab>
    if verb > 2:
        print(f"   Input ab        :  {ab}")

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
            print(f"\n>  <ab> IS {ab} WHICH IS ZERO; returning")
        else:
            print(f"   Calculated ab   :  {ab_calc}")

    return ab_calc, msrc, mrec


def check_bipole(inp, name):
    r"""Check di-/bipole parameters.

    This check-function is called from one of the modelling routines in
    :mod:`empymod.model`. Consult these modelling routines for a detailed
    description of the input parameters.

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
        inp_x = _check_var(inp[0], float, 1, name+'-x')

        # Check y and ensure it has same dimension as x
        inp_y = _check_var(inp[1], float, 1, name+'-y', inp_x.shape)

        # Check z
        inp_z = _check_var(inp[2], float, 1, name+'-z', (1,), inp_x.shape)

        # Check if all depths are the same, if so replace by one value
        if np.all(np.isclose(inp_z-inp_z[0], 0)):
            inp_z = np.array([inp_z[0]])

        return [inp_x, inp_y, inp_z]

    # Check length of inp.
    narr = len(inp)
    if narr not in [5, 6]:
        raise ValueError(f"Parameter {name} has wrong length! : "
                         f"{narr} instead of 5 (dipole) or 6 (bipole).")

    # Flag if it is a dipole or not
    isdipole = narr == 5

    if isdipole:  # dipole checks
        # Check x, y, and z
        out = chck_dipole(inp, name)

        # Check azimuth and dip
        inp_a = _check_var(inp[3], float, 1, 'azimuth', (1,), out[0].shape)
        inp_d = _check_var(inp[4], float, 1, 'dip', (1,), out[0].shape)

        # How many different depths
        nz = out[2].size

        # Expand azimuth and dip to match number of depths
        if nz > 1:
            if inp_a.size == 1:
                inp_a = np.ones(nz)*inp_a
            if inp_d.size == 1:
                inp_d = np.ones(nz)*inp_d

        out = [*out, inp_a, inp_d]

    else:         # bipole checks
        # Check each pole for x, y, and z
        out0 = chck_dipole(inp[::2], name+'-1')   # [x0, y0, z0]
        out1 = chck_dipole(inp[1::2], name+'-2')  # [x1, y1, z1]

        # If one pole has a single depth, but the other has various
        # depths, we have to repeat the single depth, as we will have
        # to loop over them.
        if out0[2].size != out1[2].size:
            if out0[2].size == 1:
                out0[2] = np.repeat(out0[2], out1[2].size)
            else:
                out1[2] = np.repeat(out1[2], out0[2].size)

        # Check if inp is a dipole instead of a bipole
        # (This is a problem, as we would could not define the angles then.)
        if not np.all((out0[0] != out1[0]) + (out0[1] != out1[1]) +
                      (out0[2] != out1[2])):
            raise ValueError(f"At least one of <{name}> is a point dipole, "
                             "use the format\n[x, y, z, azimuth, dip] "
                             "instead of [x0, x1, y0, y1, z0, z1].")

        # Collect elements
        out = [out0[0], out1[0], out0[1], out1[1], out0[2], out1[2]]

        # How many different depths
        nz = out[4].size

    return out, out[0].size, nz, isdipole


def check_dipole(inp, name, verb):
    r"""Check dipole parameters.

    This check-function is called from one of the modelling routines in
    :mod:`empymod.model`. Consult these modelling routines for a detailed
    description of the input parameters.

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
    _check_shape(np.squeeze(np.asarray(inp, dtype=object)), name, (3,))
    inp_x = _check_var(inp[0], float, 1, name+'-x')
    inp_y = _check_var(inp[1], float, 1, name+'-y', inp_x.shape)
    inp_z = _check_var(inp[2], float, 1, name+'-z', (1,))
    out = [inp_x, inp_y, inp_z]

    # Print spatial parameters
    if verb > 2:
        # Pole-type: src or rec
        if name == 'src':
            longname = '   Source(s)       : '
        else:
            longname = '   Receiver(s)     : '

        print(f"{longname} {out[0].size} dipole(s)")
        tname = ['x  ', 'y  ', 'z  ']
        for i in range(3):
            text = "     > " + tname[i] + "     [m] : "
            _prnt_min_max_val(out[i], text, verb)

    return out, out[0].size


def check_frequency(freq, res, aniso, epermH, epermV, mpermH, mpermV, verb):
    r"""Calculate frequency-dependent parameters.

    This check-function is called from one of the modelling routines in
    :mod:`empymod.model`. Consult these modelling routines for a detailed
    description of the input parameters.

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

    # As soon as at least one freq >0, we assume frequencies. Only if ALL are
    # below 0 we assume Laplace and take the negative of it.
    if np.any(freq > 0):
        laplace = False
        text_min = "Frequencies"
        text_verb = "   frequency"
    else:
        laplace = True
        freq = -freq
        text_min = "Laplace val"
        text_verb = "   s-value  "

    # Minimum frequency to avoid division by zero at freq = 0 Hz.
    # => min_freq can be set with utils.set_min
    freq = _check_min(freq, _min_freq, text_min, "Hz", verb)
    if verb > 2:
        _prnt_min_max_val(freq, text_verb+"  [Hz] : ", verb)

    # Define Laplace parameter sval.
    if laplace:
        sval = freq
    else:
        sval = 2j*np.pi*freq

    # Calculate eta and zeta (horizontal and vertical)
    c = 299792458              # Speed of light m/s
    mu_0 = 4e-7*np.pi          # Magn. permeability of free space [H/m]
    epsilon_0 = 1./(mu_0*c*c)  # Elec. permittivity of free space [F/m]

    etaH = 1/res + np.outer(sval, epermH*epsilon_0)
    etaV = 1/(res*aniso*aniso) + np.outer(sval, epermV*epsilon_0)
    zetaH = np.outer(sval, mpermH*mu_0)
    zetaV = np.outer(sval, mpermV*mu_0)

    return freq, etaH, etaV, zetaH, zetaV


def check_hankel(ht, htarg, verb):
    r"""Check Hankel transform parameters.

    This check-function is called from one of the modelling routines in
    :mod:`empymod.model`. Consult these modelling routines for a detailed
    description of the input parameters.

    Parameters
    ----------
    ht : {'dlf', 'qwe', 'quad'}
        Flag to choose the Hankel transform.

    htarg : dict
        Arguments of Hankel transform; depends on the value for `ht`.

    verb : {0, 1, 2, 3, 4}
        Level of verbosity.


    Returns
    -------
    ht, htarg
        Checked if valid and set to defaults if not provided.

    """

    # Ensure ht is all lowercase
    ht = ht.lower()

    # Initiate output dict
    targ = {}
    args = copy.deepcopy(htarg)

    if ht == 'dlf':     # DLF

        # If filter is a name (str), get it
        targ['dlf'] = args.pop('dlf', filters.key_201_2009())
        if isinstance(targ['dlf'], str):
            targ['dlf'] = getattr(filters, targ['dlf'])()

        # Ensure the provided filter has the necessary attributes.
        base = hasattr(targ['dlf'], 'base')
        j0 = hasattr(targ['dlf'], 'j0')
        j1 = hasattr(targ['dlf'], 'j1')
        factor = hasattr(targ['dlf'], 'factor')
        if not base or not j0 or not j1 or not factor:
            raise AttributeError(
                    "DLF-filter is missing some attributes; "
                    f"base: {base}; j0: {j0}; j1: {j1}; factor: {factor}.")

        # Check dimension and type of pts_per_dec
        targ['pts_per_dec'] = _check_var(
                args.pop('pts_per_dec', 0.0), float, 0, 'dlf: pts_per_dec',
                ())

        # If verbose, print Hankel transform information
        if verb > 2:
            print("   Hankel          :  DLF (Fast Hankel Transform)")
            print(f"     > Filter      :  {targ['dlf'].name}")
            pstr = "     > DLF type    :  "
            if targ['pts_per_dec'] < 0:
                print(f"{pstr}Lagged Convolution")
            elif targ['pts_per_dec'] > 0:
                print(f"{pstr}Splined, {targ['pts_per_dec']} pts/dec")
            else:
                print(f"{pstr}Standard")

    elif ht == 'qwe':   # QWE

        # rtol : 1e-12
        targ['rtol'] = _check_var(
                args.pop('rtol', 1e-12), float, 0, 'qwe: rtol', ())

        # atol : 1e-30
        targ['atol'] = _check_var(
                args.pop('atol', 1e-30), float, 0, 'qwe: atol', ())

        # nquad : 51
        targ['nquad'] = _check_var(
                args.pop('nquad', 51), int, 0, 'qwe: nquad', ())

        # maxint : 100
        targ['maxint'] = _check_var(
                args.pop('maxint', 100), int, 0, 'qwe: maxint', ())

        # pts_per_dec : 0  # No spline
        pts_per_dec = _check_var(
                args.pop('pts_per_dec', 0), int, 0, 'qwe: pts_per_dec', ())
        targ['pts_per_dec'] = _check_min(
                pts_per_dec, 0, 'pts_per_dec', '', verb)

        # diff_quad : 100
        targ['diff_quad'] = _check_var(
                args.pop('diff_quad', 100), float, 0, 'qwe: diff_quad', ())

        # a : None
        targ['a'] = args.pop('a', None)
        if targ['a'] is not None:
            targ['a'] = _check_var(targ['a'], float, 0, 'qwe: a (quad)', ())

        # b : None
        targ['b'] = args.pop('b', None)
        if targ['b'] is not None:
            targ['b'] = _check_var(targ['b'], float, 0, 'qwe: b (quad)', ())

        # limit : None
        targ['limit'] = args.pop('limit', None)
        if targ['limit'] is not None:
            targ['limit'] = _check_var(
                    targ['limit'], int, 0, 'qwe: limit (quad)', ())

        # If verbose, print Hankel transform information
        if verb > 2:
            print("   Hankel          :  Quadrature-with-Extrapolation")
            print(f"     > rtol        :  {targ['rtol']}")
            print(f"     > atol        :  {targ['atol']}")
            print(f"     > nquad       :  {targ['nquad']}")
            print(f"     > maxint      :  {targ['maxint']}")
            print(f"     > pts_per_dec :  {targ['pts_per_dec']}")
            print(f"     > diff_quad   :  {targ['diff_quad']}")
            if targ['a']:
                print(f"     > a     (quad):  {targ['a']}")
            if targ['b']:
                print(f"     > b     (quad):  {targ['b']}")
            if targ['limit']:
                print(f"     > limit (quad):  {targ['limit']}")

    elif ht in 'quad':  # QUAD

        # rtol : 1e-12
        targ['rtol'] = _check_var(
                args.pop('rtol', 1e-12), float, 0, 'quad: rtol', ())

        # atol : 1e-20
        targ['atol'] = _check_var(
                args.pop('atol', 1e-20), float, 0, 'quad: atol', ())

        # limit : 500
        targ['limit'] = _check_var(
                args.pop('limit', 500), int, 0, 'quad: limit', ())

        # a : 1e-6
        targ['a'] = _check_var(args.pop('a', 1e-6), float, 0, 'quad: a', ())

        # b : 0.1
        targ['b'] = _check_var(args.pop('b', 0.1), float, 0, 'quad: b', ())

        # pts_per_dec : 40
        pts_per_dec = _check_var(
                args.pop('pts_per_dec', 40), int, 0, 'quad: pts_per_dec', ())
        targ['pts_per_dec'] = _check_min(
                pts_per_dec, 1, 'pts_per_dec', '', verb)

        # If verbose, print Hankel transform information
        if verb > 2:
            print("   Hankel          :  Quadrature")
            print(f"     > rtol        :  {targ['rtol']}")
            print(f"     > atol        :  {targ['atol']}")
            print(f"     > limit       :  {targ['limit']}")
            print(f"     > a           :  {targ['a']}")
            print(f"     > b           :  {targ['b']}")
            print(f"     > pts_per_dec :  {targ['pts_per_dec']}")

    else:
        raise ValueError("<ht> must be one of: ['dlf', 'qwe', 'quad'];"
                         f" <ht> provided: {ht}.")

    # Check remaining arguments.
    if args and verb > 0:
        print(f"* WARNING :: Unknown htarg {args} for method '{ht}'")

    return ht, targ


def check_model(depth, res, aniso, epermH, epermV, mpermH, mpermV, xdirect,
                verb):
    r"""Check the model: depth and corresponding layer parameters.

    This check-function is called from one of the modelling routines in
    :mod:`empymod.model`. Consult these modelling routines for a detailed
    description of the input parameters.

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

    # If all depths are decreasing, swap depth and parameters.
    if depth.size > 1 and np.all(depth[1:] - depth[:-1] < 0):
        swap = -1
    else:
        swap = 1
    depth = depth[::swap]

    # Ensure depth is increasing
    if np.any(depth[1:] - depth[:-1] < 0):
        raise ValueError(f"Depth must be continuously increasing or decreasing"
                         f".\n<depth> provided: {_strvar(depth[::swap])}.")

    # Add -infinity at the beginning
    # => The top-layer (-infinity to first interface) is layer 0.
    if depth.size == 0:
        depth = np.array([-np.infty, ])
    else:
        if depth[0] != -np.infty:
            depth = np.r_[-np.infty, depth]

        # Remove +np.infty (can be used to define 2-layer coordinate system).
        if depth[-1] == np.infty:
            depth = depth[:-1]

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
    # We assume isotropic behaviour if epermH was provided but not epermV
    if epermV is None:
        epermV = epermH
    else:
        epermV = check_inp(epermV, 'epermV', 0.0)
    mpermH = check_inp(mpermH, 'mpermH', 0.0)
    # We assume isotropic behaviour if mpermH was provided but not mpermV
    if mpermV is None:
        mpermV = mpermH
    else:
        mpermV = check_inp(mpermV, 'mpermV', 0.0)

    # Swap parameters if depths were given in reverse.
    res = res[::swap]
    aniso = aniso[::swap]
    epermH = epermH[::swap]
    epermV = epermV[::swap]
    mpermH = mpermH[::swap]
    mpermV = mpermV[::swap]

    # Print model parameters
    if verb > 2:
        print(f"   depth       [m] :  {_strvar(depth[1:])}")
        print(f"   res     [Ohm.m] :  {_strvar(res)}")
        print(f"   aniso       [-] :  {_strvar(aniso)}")
        print(f"   epermH      [-] :  {_strvar(epermH)}")
        print(f"   epermV      [-] :  {_strvar(epermV)}")
        print(f"   mpermH      [-] :  {_strvar(mpermH)}")
        print(f"   mpermV      [-] :  {_strvar(mpermV)}")

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
            print("\n>  MODEL IS A FULLSPACE; returning analytical "
                  "frequency-domain solution")
        else:
            print("\n>  MODEL IS A FULLSPACE")

    # Print xdirect info
    if verb > 2:
        if xdirect is None:
            print("   direct field    :  Not calculated (secondary field)")
        elif xdirect:
            print("   direct field    :  Comp. in frequency domain")
        else:
            print("   direct field    :  Comp. in wavenumber domain")

    return depth, res, aniso, epermH, epermV, mpermH, mpermV, isfullspace


def check_loop(loop, ht, htarg, verb):
    r"""Check loop parameter.

    This check-function is called from one of the modelling routines in
    :mod:`empymod.model`. Consult these modelling routines for a detailed
    description of the input parameters.

    Parameters
    ----------
    loop : {None, 'freq', 'off'}
        Loop flag.

    ht : {'dlf', 'qwe', 'quad'}
        Flag to choose the Hankel transform.

    htarg : dict
        Arguments of Hankel transform; depends on the value for `ht`.

    verb : {0, 1, 2, 3, 4}
        Level of verbosity.


    Returns
    -------
    loop_freq : bool
        Boolean if to loop over frequencies.

    loop_off : bool
        Boolean if to loop over offsets.

    """

    # Define if to loop over frequencies or over offsets
    lagged_splined_dlf = False
    if ht == 'dlf':
        if htarg['pts_per_dec'] != 0:
            lagged_splined_dlf = True

    if ht in ['qwe', 'quad'] or lagged_splined_dlf:
        loop_freq = True
        loop_off = False
    else:
        loop_off = loop == 'off'
        loop_freq = loop == 'freq'

    # If verbose, print loop information
    if verb > 2:

        if loop_off:
            print("   Loop over       :  Offsets")
        elif loop_freq:
            print("   Loop over       :  Frequencies")
        else:
            print("   Loop over       :  None (all vectorized)")

    return loop_freq, loop_off


def check_time(time, signal, ft, ftarg, verb):
    r"""Check time domain specific input parameters.

    This check-function is called from one of the modelling routines in
    :mod:`empymod.model`. Consult these modelling routines for a detailed
    description of the input parameters.

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

    ft : {'dlf', 'qwe', 'fftlog', 'fft'}
        Flag for Fourier transform.

    ftarg : dict
        Arguments of Fourier transform; depends on the value for `ft`.

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

    # Initiate output dict
    targ = {}
    args = copy.deepcopy(ftarg)

    if ft == 'dlf':       # Fourier-DLF (sin/cos-filters)

        # Check dimension and type of pts_per_dec
        targ['pts_per_dec'] = _check_var(
                args.pop('pts_per_dec', -1.0), float, 0, 'dlf: pts_per_dec',
                ())

        # Check kind; if switch-off/on is required, ensure kind is cosine/sine
        targ['kind'] = args.pop('kind', 'sin')
        if signal > 0:
            targ['kind'] = 'sin'
        elif signal < 0:
            targ['kind'] = 'cos'
        if targ['kind'] not in ['sin', 'cos']:
            raise ValueError("'kind' must be either 'sin' or 'cos'; "
                             f"provided: {targ['kind']}.")

        # If filter is a name (str), get it
        targ['dlf'] = args.pop('dlf', filters.key_201_CosSin_2012())
        if isinstance(targ['dlf'], str):
            targ['dlf'] = getattr(filters, targ['dlf'])()

        # Ensure the provided filter has the necessary attributes.
        base = hasattr(targ['dlf'], 'base')
        if targ['kind'] == 'sin':
            sincos = hasattr(targ['dlf'], 'sin')
        else:
            sincos = hasattr(targ['dlf'], 'cos')
        factor = hasattr(targ['dlf'], 'factor')
        if not base or not sincos or not factor:
            raise AttributeError(
                    "DLF-filter is missing some attributes; base: "
                    f"{base}; {targ['kind']}: {sincos}; factor: {factor}.")

        # If verbose, print Fourier transform information
        if verb > 2:
            if targ['kind'] == 'sin':
                print("   Fourier         :  DLF (Sine-Filter)")
            else:
                print("   Fourier         :  DLF (Cosine-Filter)")
            print(f"     > Filter      :  {targ['dlf'].name}")
            pstr = "     > DLF type    :  "
            if targ['pts_per_dec'] < 0:
                print(f"{pstr}Lagged Convolution")
            elif targ['pts_per_dec'] > 0:
                print(f"{pstr}Splined, {targ['pts_per_dec']} pts/dec")
            else:
                print(f"{pstr}Standard")

        # Get required frequencies
        omega, _ = transform.get_dlf_points(
                targ['dlf'], time, targ['pts_per_dec'])
        freq = np.squeeze(omega/2/np.pi)

    elif ft == 'qwe':     # QWE (using sine and imag-part)

        # If switch-off is required, use cosine, else sine
        args.pop('sincos', None)
        if signal >= 0:
            targ['sincos'] = np.sin
        else:
            targ['sincos'] = np.cos

        # rtol : 1e-8
        targ['rtol'] = _check_var(
                args.pop('rtol', 1e-8), float, 0, 'qwe: rtol', ())

        # atol : 1e-20
        targ['atol'] = _check_var(
                args.pop('atol', 1e-20), float, 0, 'qwe: atol', ())

        # nquad : 21
        targ['nquad'] = _check_var(
                args.pop('nquad', 21), int, 0, 'qwe: nquad', ())

        # maxint : 200
        targ['maxint'] = _check_var(
                args.pop('maxint', 200), int, 0, 'qwe: maxint', ())

        # pts_per_dec : 20
        pts_per_dec = _check_var(
                args.pop('pts_per_dec', 20), int, 0, 'qwe: pts_per_dec', ())
        targ['pts_per_dec'] = _check_min(
                pts_per_dec, 1, 'pts_per_dec', '', verb)

        # diff_quad : 100
        targ['diff_quad'] = _check_var(
                args.pop('diff_quad', 100), int, 0, 'qwe: diff_quad', ())

        # a : None
        targ['a'] = args.pop('a', None)
        if targ['a'] is not None:
            targ['a'] = _check_var(targ['a'], float, 0, 'qwe: a (quad)', ())

        # b : None
        targ['b'] = args.pop('b', None)
        if targ['b'] is not None:
            targ['b'] = _check_var(targ['b'], float, 0, 'qwe: b (quad)', ())

        # limit : None
        targ['limit'] = args.pop('limit', None)
        if targ['limit'] is not None:
            targ['limit'] = _check_var(
                    targ['limit'], int, 0, 'qwe: limit (quad)', ())

        # If verbose, print Fourier transform information
        if verb > 2:
            print("   Fourier         :  Quadrature-with-Extrapolation")
            print(f"     > rtol        :  {targ['rtol']}")
            print(f"     > atol        :  {targ['atol']}")
            print(f"     > nquad       :  {targ['nquad']}")
            print(f"     > maxint      :  {targ['maxint']}")
            print(f"     > pts_per_dec :  {targ['pts_per_dec']}")
            print(f"     > diff_quad   :  {targ['diff_quad']}")
            if targ['a']:
                print(f"     > a     (quad):  {targ['a']}")
            if targ['b']:
                print(f"     > b     (quad):  {targ['b']}")
            if targ['limit']:
                print(f"     > limit (quad):  {targ['limit']}")

        # Get required frequencies
        g_x, _ = special.roots_legendre(targ['nquad'])
        minf = np.floor(10*np.log10((g_x.min() + 1)*np.pi/2/time.max()))/10
        maxf = np.ceil(10*np.log10(targ['maxint']*np.pi/time.min()))/10
        freq = np.logspace(minf, maxf, int((maxf-minf)*targ['pts_per_dec']+1))

    elif ft == 'fftlog':  # FFTLog (using sine and imag-part)

        # pts_per_dec : 10
        pts_per_dec = _check_var(
                args.pop('pts_per_dec', 10), int, 0,
                'fftlog: pts_per_dec', ())
        targ['pts_per_dec'] = _check_min(
                pts_per_dec, 1, 'pts_per_dec', '', verb)

        # add_dec : [-2, 1]
        targ['add_dec'] = _check_var(
                args.pop('add_dec', np.array([-2, 1])), float, 1,
                'fftlog: add_dec', (2,))

        # q : 0
        targ['q'] = _check_var(args.pop('q', 0), float, 0, 'fftlog: q', ())
        # Restrict q to +/- 1
        if np.abs(targ['q']) > 1:
            targ['q'] = np.sign(targ['q'])

        # If switch-off is required, use cosine, else sine
        args.pop('mu', None)
        if signal >= 0:
            targ['mu'] = 0.5
        else:
            targ['mu'] = -0.5

        # If verbose, print Fourier transform information
        if verb > 2:
            print("   Fourier         :  FFTLog")
            print(f"     > pts_per_dec :  {targ['pts_per_dec']}")
            print(f"     > add_dec     :  {targ['add_dec']}")
            print(f"     > q           :  {targ['q']}")

        # Calculate minimum and maximum required frequency
        minf = np.log10(1/time.max()) + targ['add_dec'][0]
        maxf = np.log10(1/time.min()) + targ['add_dec'][1]
        n = np.int64(maxf - minf)*targ['pts_per_dec']

        # Initialize FFTLog, get required parameters
        freq, tcalc, dlnr, kr, rk = transform.get_fftlog_input(
                minf, maxf, n, targ['q'], targ['mu'])
        targ['tcalc'] = tcalc
        targ['dlnr'] = dlnr
        targ['kr'] = kr
        targ['rk'] = rk
        for name in ['tcalc', 'dlnr', 'kr', 'rk']:
            # So they don't get caught in the args-check.
            args.pop(name, None)

    elif ft == 'fft':     # FFT
        # Keys: dfreq, nfreq, ntot, pts_per_dec, fftfreq

        # dfreq : 0.002
        targ['dfreq'] = _check_var(
                args.pop('dfreq', 0.002), float, 0, 'fft: dfreq', ())

        # nfreq : 2048
        targ['nfreq'] = _check_var(
                args.pop('nfreq', 2048), int, 0, 'fft: nfreq', ())

        # ntot
        nall = 2**np.arange(30)
        targ['ntot'] = _check_var(
            args.pop('ntot', nall[np.argmax(nall >= targ['nfreq'])]),  # (*)
            int, 0, 'fft: ntot', ())
        # Assure that input ntot is not bigger than nfreq
        if targ['nfreq'] > targ['ntot']:
            targ['ntot'] = nall[np.argmax(nall >= targ['nfreq'])]
        # (*) We could use here fftpack.next_fast_len, but tests have shown
        #     that powers of two yield better results in this case.

        # pts_per_dec : None
        targ['pts_per_dec'] = args.pop('pts_per_dec', None)
        if targ['pts_per_dec'] is not None:
            pts_per_dec = _check_var(
                    targ['pts_per_dec'], int, 0, 'fft: pts_per_dec', ())
            targ['pts_per_dec'] = _check_min(
                    pts_per_dec, 1, 'pts_per_dec', '', verb)

        # Get required frequencies
        if targ['pts_per_dec']:  # Space actually calc. freqs logarithmically.
            start = np.log10(targ['dfreq'])
            stop = np.log10(targ['nfreq']*targ['dfreq'])
            freq = np.logspace(
                    start, stop, int((stop-start)*targ['pts_per_dec'] + 1))
        else:
            freq = np.arange(1, targ['nfreq']+1)*targ['dfreq']

        # If verbose, print Fourier transform information
        if verb > 2:
            print("   Fourier         :  Fast Fourier Transform FFT")
            print(f"     > dfreq       :  {targ['dfreq']}")
            print(f"     > nfreq       :  {targ['nfreq']}")
            print(f"     > ntot        :  {targ['ntot']}")
            if targ['pts_per_dec']:
                print(f"     > pts_per_dec :  {targ['pts_per_dec']}")
            else:
                print("     > pts_per_dec :  (linear)")

    else:
        raise ValueError("<ft> must be one of: ['dlf', 'qwe', "
                         f"'fftlog', 'fft']; <ft> provided: {ft}")

    # Check remaining arguments.
    if args and verb > 0:
        print(f"* WARNING :: Unknown ftarg {args} for method '{ft}'")

    return time, freq, ft, targ


def check_time_only(time, signal, verb):
    r"""Check time and signal parameters.

    This check-function is called from one of the modelling routines in
    :mod:`empymod.model`. Consult these modelling routines for a detailed
    description of the input parameters.

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
        raise ValueError("<signal> must be one of: [None, -1, 0, 1]; "
                         f"<signal> provided: {signal}")

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
    :mod:`empymod.model`. Consult these modelling routines for a detailed
    description of the input parameters.

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
        raise ValueError(
                "Solution must be one of ['fs', 'dfs', 'dhs', "
                f"'dsplit', 'dtetm']; <solution> provided: {solution}")

    # If diffusive solution is required, ensure EE-field.
    if solution[0] == 'd' and (msrc or mrec):
        raise ValueError(
                "Diffusive solution is only implemented for electric "
                f"sources and electric receivers, <ab> provided: {ab}")

    # If full solution is required, ensure frequency-domain.
    if solution == 'fs' and signal is not None:
        raise ValueError(
                "Full fullspace solution is only implemented for "
                f"the frequency domain, <signal> provided: {signal}")


# 2.b <Get>s (alphabetically)

def get_abs(msrc, mrec, srcazm, srcdip, recazm, recdip, verb):
    r"""Get required ab's for given angles.

    This check-function is called from one of the modelling routines in
    :mod:`empymod.model`. Consult these modelling routines for a detailed
    description of the input parameters.

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
    bab = np.asarray(ab_calc*0+1, dtype=np.bool_)

    # Remove if source is x- or y-directed
    check = np.atleast_1d(srcazm)
    if np.allclose(srcazm % (np.pi/2), 0):  # if all angles are multiples of 90
        if np.all(np.isclose(check // (np.pi/2) % 2, 0)):
            # Multiples of pi (180)
            bab[:, 1] *= False        # x-directed source, remove y
        elif np.all(np.isclose(check // (np.pi/2) % 2, 1)):
            # Multiples of pi/2 (90)
            bab[:, 0] *= False        # y-directed source, remove x

    # Remove if source is vertical
    check = np.atleast_1d(srcdip)
    if np.allclose(srcdip % (np.pi/2), 0):  # if all angles are multiples of 90
        if np.all(np.isclose(check // (np.pi/2) % 2, 0)):
            # Multiples of pi (180)
            bab[:, 2] *= False        # Horizontal, remove z
        elif np.all(np.isclose(check // (np.pi/2) % 2, 1)):
            # Multiples of pi/2 (90)
            bab[:, :2] *= False       # Vertical, remove x/y

    # Remove if receiver is x- or y-directed
    check = np.atleast_1d(recazm)
    if np.allclose(recazm % (np.pi/2), 0):  # if all angles are multiples of 90
        if np.all(np.isclose(check // (np.pi/2) % 2, 0)):
            # Multiples of pi (180)
            bab[1, :] *= False        # x-directed receiver, remove y
        elif np.all(np.isclose(check // (np.pi/2) % 2, 1)):
            # Multiples of pi/2 (90)
            bab[0, :] *= False        # y-directed receiver, remove x

    # Remove if receiver is vertical
    check = np.atleast_1d(recdip)
    if np.allclose(recdip % (np.pi/2), 0):  # if all angles are multiples of 90
        if np.all(np.isclose(check // (np.pi/2) % 2, 0)):
            # Multiples of pi (180)
            bab[2, :] *= False        # Horizontal, remove z
        elif np.all(np.isclose(check // (np.pi/2) % 2, 1)):
            # Multiples of pi/2 (90)
            bab[:2, :] *= False       # Vertical, remove x/y

    # Reduce
    ab_calc = ab_calc[bab].ravel()

    # Print actual calculated <ab>
    if verb > 2:
        print(f"   Required ab's   :  {_strvar(ab_calc)}")

    return ab_calc


def get_geo_fact(ab, srcazm, srcdip, recazm, recdip, msrc, mrec):
    r"""Get required geometrical scaling factor for given angles.

    This check-function is called from one of the modelling routines in
    :mod:`empymod.model`. Consult these modelling routines for a detailed
    description of the input parameters.

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
    fact = np.outer(frec, fsrc)

    # Set very small angles to proper zero (because e.g. sin(pi/2) != exact 0)
    # => min_angle can be set with utils.set_min
    fact[np.abs(fact) < _min_angle] = 0

    return fact


def get_layer_nr(inp, depth):
    r"""Get number of layer in which inp resides.

    .. note::

        If zinp is on a layer interface, the layer above the interface is
        chosen.

    This check-function is called from one of the modelling routines in
    :mod:`empymod.model`. Consult these modelling routines for a detailed
    description of the input parameters.

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
    zinp = np.array(inp[2], dtype=np.float64)

    #  depth = [-infty : last interface]; create additional depth-array
    # pdepth = [fist interface : +infty]
    pdepth = np.concatenate((depth[1:], np.array([np.infty])))

    # Broadcast arrays
    b_zinp = np.atleast_1d(zinp)[:, None]

    # Get layers
    linp = np.where((depth[None, :] < b_zinp)*(pdepth[None, :] >= b_zinp))[1]

    # Return; squeeze in case of only one inp-depth
    return np.squeeze(linp), np.squeeze(zinp)


def get_off_ang(src, rec, nsrc, nrec, verb):
    r"""Get depths, offsets, angles, hence spatial input parameters.

    This check-function is called from one of the modelling routines in
    :mod:`empymod.model`. Consult these modelling routines for a detailed
    description of the input parameters.

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
    :mod:`empymod.model`. Consult these modelling routines for a detailed
    description of the input parameters.

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

        - If 0, output is normalized to source and receiver of 1 m length, and
          source strength of 1 A.
        - If != 0, output is returned for given source and receiver length, and
          source strength.

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
                    np.atleast_1d(inp[2][iz]), np.atleast_1d(inp[3][iz]),
                    np.atleast_1d(inp[4][iz])]
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

        # If dipole, inp_w are ones, unless strength > 0
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
        dl = np.atleast_1d(np.linalg.norm(
            np.array([dx, dy, dz], dtype=object), axis=0))

        # Horizontal deviation from x-axis
        azm = np.atleast_1d(np.arctan2(dy, dx))

        # Vertical deviation from xy-plane down
        dip = np.atleast_1d(np.pi/2-np.arccos(dz/dl))

        # Check intpts
        intpts = _check_var(intpts, int, 0, 'intpts', ())

        # Gauss quadrature if intpts > 2; else set to center of tinp
        if intpts > 2:  # Calculate the dipole positions
            # Get integration positions and weights
            g_x, g_w = special.roots_legendre(intpts)
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
            print(f"{longname} {tout[0].size} dipole(s)")
            tname = ['x  ', 'y  ', 'z  ']
            prntinp = tout
        else:
            print(f"{longname} {int(tout[0].size/intpts)} bipole(s)")
            tname = ['x_c', 'y_c', 'z_c']
            if intpts == 1:
                print("     > intpts      :  1 (as dipole)")
                prntinp = tout
            else:
                print(f"     > intpts      :  {intpts}")
                prntinp = [np.atleast_1d(tinp[0])[0] + dx/2,
                           np.atleast_1d(tinp[2])[0] + dy/2,
                           np.atleast_1d(tinp[4])[0] + dz/2]

            # Print bipole length and strength
            _prnt_min_max_val(dl, "     > length  [m] : ", verb)
            print(f"     > strength[A] :  {_strvar(strength)}")

        # Print coordinates
        for i in range(3):
            text = "     > " + tname[i] + "     [m] : "
            _prnt_min_max_val(prntinp[i], text, verb)

        # Print angles
        _prnt_min_max_val(np.rad2deg(azm), "     > azimuth [°] : ", verb)
        _prnt_min_max_val(np.rad2deg(dip), "     > dip     [°] : ", verb)

    return tout, azm, dip, g_w, intpts, inp_w


def get_kwargs(names, defaults, kwargs):
    """Return wanted parameters, check remaining.

    1. Extracts parameters `names` from `kwargs`, filling them with the
       `defaults`-value if it is not in `kwargs`.
    2. Check remaining kwargs;

       - Raise an error if it is an unknown keyword;
       - Print warning if it is a keyword from another routine (verb>0).

    List of possible kwargs:

    - ALL functions: src, rec, res, aniso, epermH, epermV, mpermH, mpermV, verb
    - ONLY gpr: cf, gain
    - ONLY bipole: msrc, srcpts
    - ONLY dipole_k: freq, wavenumber
    - ONLY analytical: solution
    - ONLY bipole, loop: mrec, recpts, strength
    - ONLY bipole, dipole, loop, gpr: ht, htarg, ft, ftarg, xdirect, loop
    - ONLY bipole, dipole, loop, analytical: signal, squeeze
    - ONLY dipole, analytical, gpr, dipole_k: ab
    - ONLY bipole, dipole, loop, gpr, dipole_k: depth
    - ONLY bipole, dipole, loop, analytical, gpr: freqtime

    Parameters
    ----------
    names: list
        Names of wanted parameters as strings.

    defaults: list
        Default values of wanted parameters, in same order.

    kwargs : dict
        Passed-through kwargs.

    Returns
    ------
    values : list
        Wanted parameters.

    """
    # Known keys (excludes keys present in ALL routines).
    known_keys = set([
            'depth', 'ht', 'htarg', 'ft', 'ftarg', 'xdirect', 'loop', 'signal',
            'ab', 'freqtime', 'freq', 'wavenumber', 'solution', 'cf', 'gain',
            'msrc', 'srcpts', 'mrec', 'recpts', 'strength', 'squeeze'
    ])

    # Loop over wanted parameters.
    out = list()
    verb = 2  # get_kwargs-internal default.
    for i, name in enumerate(names):

        # Catch verb for warnings later on.
        if name == 'verb':
            verb = kwargs.get(name, defaults[i])

        # Add this parameter to the list.
        out.append(kwargs.pop(name, defaults[i]))

    # Check remaining parameters.
    if kwargs:
        if not set(kwargs.keys()).issubset(known_keys):
            raise TypeError(f"Unexpected **kwargs: {kwargs}.")
        elif verb > 0:
            print(f"* WARNING :: Unused **kwargs: {kwargs}.")

    return out


def printstartfinish(verb, inp=None, kcount=None):
    r"""Print start and finish with time measure and kernel count."""
    if inp:
        if verb > 1:
            ttxt = str(timedelta(seconds=default_timer() - inp))
            ktxt = ' '
            if kcount:
                ktxt += str(kcount) + ' kernel call(s)'
            print(f"\n:: empymod END; runtime = {ttxt} ::{ktxt}\n")
    else:
        t0 = default_timer()
        if verb > 2:
            print(f"\n:: empymod START  ::  v{__version__}\n")
        return t0


def conv_warning(conv, targ, name, verb):
    r"""Print error if QWE/QUAD did not converge at least once."""
    if verb > 0 and not conv:
        print(f"* WARNING :: {name}"
              "-quadrature did not converge at least once;\n             "
              "=> desired `atol` and `rtol` might not be achieved.")


# 3. Set/get min values

def set_minimum(min_freq=None, min_time=None, min_off=None, min_res=None,
                min_angle=None):
    r"""
    Set minimum values of parameters.

    The given parameters are set to its minimum value if they are smaller.

    .. note::

        set_minimum and get_minimum are derived after set_printoptions and
        get_printoptions from arrayprint.py in numpy.

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

    .. note::

        set_minimum and get_minimum are derived after set_printoptions and
        get_printoptions from arrayprint.py in numpy.

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
                raise ValueError(f"Parameter {name} has wrong shape! : "
                                 f"{varshape} instead of {shape} or {shape2}.")
        else:
            raise ValueError(f"Parameter {name} has wrong shape! : "
                             f"{varshape} instead of {shape}.")


def _check_var(var, dtype, ndmin, name, shape=None, shape2=None):
    r"""Return variable as array of dtype, ndmin; shape-checked."""
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
        print(f"{text} {_strvar(var.min())} - {_strvar(var.max())} "
              f": {_strvar(var.size)}  [min-max; #]")
        if verb > 3:
            print(f"                   :  {_strvar(var)}")
    else:
        print(f"{text} {_strvar(np.atleast_1d(var))}")


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
            print(f"* WARNING :: {name} < {str(minval)} {unit}"
                  f" are set to {minval} {unit}!")
    if scalar:
        return np.squeeze(par)
    else:
        return par


# 5. Report
class Report(ScoobyReport):
    r"""Print date, time, and version information.

    Use `scooby` to print date, time, and package version information in any
    environment (Jupyter notebook, IPython console, Python console, QT
    console), either as html-table (notebook) or as plain text (anywhere).

    Always shown are the OS, number of CPU(s), `numpy`, `scipy`, `numba`,
    `empymod`, `sys.version`, and time/date.

    Additionally shown are, if they can be imported, `IPython`, and
    `matplotlib`. It also shows MKL information, if available.

    All modules provided in `add_pckg` are also shown.

    .. note::

        The package `scooby` has to be installed in order to use `Report`:
        ``pip install scooby``.


    Parameters
    ----------
    add_pckg : packages, optional
        Package or list of packages to add to output information (must be
        imported beforehand).

    ncol : int, optional
        Number of package-columns in html table (no effect in text-version);
        Defaults to 3.

    text_width : int, optional
        The text width for non-HTML display modes

    sort : bool, optional
        Sort the packages when the report is shown


    Examples
    --------
    >>> import pytest
    >>> import dateutil
    >>> from empymod import Report
    >>> Report()                            # Default values
    >>> Report(pytest)                      # Provide additional package
    >>> Report([pytest, dateutil], ncol=5)  # Set nr of columns

    """

    def __init__(self, add_pckg=None, ncol=3, text_width=80, sort=False):
        """Initiate a scooby.Report instance."""

        # Mandatory packages.
        core = ['numpy', 'scipy', 'numba', 'empymod']

        # Optional packages.
        optional = ['IPython', 'matplotlib']

        super().__init__(additional=add_pckg, core=core, optional=optional,
                         ncol=ncol, text_width=text_width, sort=sort)
