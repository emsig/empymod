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
you have to call `check_ab`, `check_model`, `get_coords`, `check_depth` and
`check_hankel` only once, but `check_frequency` in each loop. You do not have
to run these checks if you are sure your input parameters are in the correct
format.

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
from scipy import special
from datetime import timedelta
from timeit import default_timer
from scipy.constants import mu_0       # Magn. permeability of free space [H/m]
from scipy.constants import epsilon_0  # Elec. permittivity of free space [F/m]

from . import filters, transform


# TODO make restrictions
# __all__ = ['EMArray', 'check_ab', 'get_abs_srcbipole', 'check_model',
#            'check_pole', 'get_coords', 'check_depth', 'check_hankel',
#            'check_frequency', 'check_opt', 'check_time', 'printstartfinish', ]

# 0. Settings

min_freq = 1e-20  # Minimum frequency  [Hz]
min_time = 1e-20  # Minimum time       [s]
min_off = 1e-3    # Minimum offset     [m]
                  # > Also used to round src- and rec-coordinates (1e-3 => mm)


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

def check_ab(ab, verb):
    """Check source-receiver configuration.

    This check-function is called from one of the modelling routines in
    :mod:`model`.  Consult these modelling routines for a detailed description
    of the input parameters.

    Parameters
    ----------
    ab : int
        Source-receiver configuration.

    verb : {0, 1, 2}
        Level of verbosity.

    Returns
    -------
    ab_calc : int
        Adjusted source-receiver configuration using reciprocity.

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
        print("   Input ab      : ", ab)

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
        if ab_calc in [36, ]:
            print("\n>  <ab> IS "+str(ab_calc)+" WHICH IS ZERO; returning")
        else:
            print("   Calc. ab      : ", ab_calc)

    return ab_calc, msrc, mrec


def check_ab_tmp(ab, verb):
    """Check source-receiver configuration.

    This check-function is called from one of the modelling routines in
    :mod:`model`.  Consult these modelling routines for a detailed description
    of the input parameters.

    Parameters
    ----------
    ab : int
        Source-receiver configuration.

    verb : {0, 1, 2}
        Level of verbosity.

    Returns
    -------
    ab_calc : int
        Adjusted source-receiver configuration using reciprocity.

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
        print("   Input ab      : ", ab)

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
        if ab_calc in [36, ]:
            print("\n>  <ab> IS "+str(ab_calc)+" WHICH IS ZERO; returning")
        else:
            print("   Calc. ab      : ", ab_calc)

    return np.array([ab_calc]), msrc, mrec


def get_abs_srcbipole(msrc, recdir, theta, phi, verb):
    """Check source-receiver configuration.

    This check-function is called from one of the modelling routines in
    :mod:`model`.  Consult these modelling routines for a detailed description
    of the input parameters.

    Parameters
    ----------
    msrc : bool
        True if src is magnetic, else False.
    recdir : {1, 2, 3, 4, 5, 6}
        Receiver direction.
    theta : float
        Horizontal source angle.
    phi : float
        Vertical source angle.
    verb : {0, 1, 2}
        Level of verbosity.


    Returns
    -------
    ab : array of int
        ab's to calculate for this bipole.
    mrec : bool
        Deduced from recdir. Receiver is magnetic if True.
    fact : array
        Geometrical spreading factors for ab's.

    """
    # Try to cast recdir into an integer
    try:
        int(recdir)
    except(TypeError, ValueError):
        print('* ERROR   :: <recdir> must be an integer')
        raise

    # Check src and rec orientation (<ab> for alpha-beta)
    # pab: all possible values that <ab> can take
    pab = [1, 2, 3, 4, 5, 6]
    if recdir not in pab:
        print('* ERROR   :: <recdir> must be one of: ' + str(pab) + ';' +
              ' <recdir> provided: ' + str(recdir))
        raise ValueError('recdir')

    # Get required ab's
    ab = 10*recdir + np.arange(1, 4)
    if msrc:
        ab += 3

    # If rec is magnetic, change ab with reciprocity (see `check_ab`)
    if recdir < 4:
        mrec = False
    else:
        if msrc:
            ab -= 33  # -30 : mrec->erec; -3: msrc->esrc
        else:
            ab = ab % 10*10 + ab // 10  # Swap alpha/beta
        mrec = True

    # => Geometrical scaling
    fact = [np.cos(theta)*np.cos(phi), np.sin(theta)*np.cos(phi), np.sin(phi)]

    # Remove unnecessary ab's
    if np.any(np.isclose(theta, [0, np.pi])):  # x-directed source, remove y
        ab = ab[::2]
        fact = fact[::2]
    elif np.any(np.isclose(theta, [np.pi/2., 3*np.pi/2.])):  # y-dir. src rem x
        ab = ab[1:]
        fact = fact[1:]
    if np.isclose(phi, 0):  # Horizontal source, remove z
        ab = ab[:-1]
        fact = fact[:-1]

    # Print actual calculated <ab>
    if verb > 1:
        print("   Required ab's : ", _strvar(ab))

    return ab, mrec, fact


def get_abs(msrc, mrec, srctheta, srcphi, rectheta, recphi, verb):
    """Check source-receiver configuration.

    This check-function is called from one of the modelling routines in
    :mod:`model`.  Consult these modelling routines for a detailed description
    of the input parameters.

    Parameters
    ----------
    msrc, mrec : bool
        True if src/rec is magnetic, else False.
    srctheta, rectheta : float
        Horizontal source/receiver angle.
    srcphi, recphi : float
        Vertical source/receiver angle.
    verb : {0, 1, 2}
        Level of verbosity.


    Returns
    -------
    ab_calc : array of int
        ab's to calculate for this bipole.
    fact : array
        Geometrical spreading factors for ab's.

    """
    # Get required ab's
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

    # => Geometrical scaling
    srcfact = [np.cos(srctheta)*np.cos(srcphi),
            np.sin(srctheta)*np.cos(srcphi), np.sin(srcphi)]
    recfact = [np.cos(rectheta)*np.cos(recphi),
            np.sin(rectheta)*np.cos(recphi), np.sin(recphi)]
    fact = np.outer(srcfact, recfact)

    # Remove unnecessary ab's and fact's
    bab = np.asarray(ab_calc*0+1, dtype=bool)

    # Remove regarding source alignment
    if np.any(np.isclose(srctheta, [0, np.pi])):  # x-directed, remove y
        bab[:, 1] *= False
    elif np.any(np.isclose(srctheta, [np.pi/2., 3*np.pi/2.])):  # y-dir, rem. x
        bab[:, 0] *= False
    if np.any(np.isclose(srcphi, [0, np.pi])):  # Horizontal, remove z
        bab[:, 2] *= False
    elif np.any(np.isclose(srcphi, [np.pi/2., 3*np.pi/2.])):  # Vert. rem. x/y
        bab[:, :2] *= False

    # Remove regarding receiver alignment
    if np.any(np.isclose(rectheta, [0, np.pi])):  # x-directed rec, remove y
        bab[1, :] *= False
    elif np.any(np.isclose(rectheta, [np.pi/2., 3*np.pi/2.])):  # y-dir, rem. x
        bab[0, :] *= False
    if np.any(np.isclose(recphi, [0, np.pi])):  # Horizontal, remove z
        bab[2, :] *= False
    elif np.any(np.isclose(recphi, [np.pi/2., 3*np.pi/2.])):  # Vert. rem. x/y
        bab[:2, :] *= False

    # Reduce
    ab_calc = ab_calc[bab].ravel()
    fact = fact[bab].ravel()

    print(ab_calc)
    print(fact)
    print(bab)

    # Print actual calculated <ab>
    if verb > 1:
        print("   Required ab's : ", _strvar(ab_calc))

    return ab_calc, fact


def check_model(depth, res, aniso, epermH, epermV, mpermH, mpermV, verb):
    """Check the model: depth and corresponding layer parameters.

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

    epermH : array_like
        Horizontal electric permittivities epsilon_h (-); #epermH = #res.

    epermV : array_like
        Vertical electric permittivities epsilon_v (-); #epermV = #res.

    mpermH : array_like
        Horizontal magnetic permeabilities mu_h (-); #mpermH = #res.

    mpermV : array_like
        Vertical magnetic permeabilities mu_v (-); #mpermV = #res.

    verb : {0, 1, 2}
        Level of verbosity.


    Returns
    -------
    depth : array
        Depths of layer interfaces, adds -infty at beginning if not present.

    res : array
        As input, checked for size.

    aniso : array
        As input, checked for size. If None, defaults to an array of ones.

    epermH : array
        As input, checked for size. If None, defaults to an array of ones.

    epermV : array
        As input, checked for size. If None, defaults to an array of ones.

    mpermH : array
        As input, checked for size. If None, defaults to an array of ones.

    mpermV : array
        As input, checked for size. If None, defaults to an array of ones.

    isfullspace : bool
        If True, the model is a fullspace (res, aniso, epermH, epermV, mpermM,
        and mpermV are in all layers the same).

    """

    # Check depth
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
        print("   depth     [m] : ", _strvar(depth[1:]))
        print("   res   [Ohm.m] : ", _strvar(res))
        print("   aniso     [-] : ", _strvar(aniso))
        print("   epermH    [-] : ", _strvar(epermH))
        print("   epermV    [-] : ", _strvar(epermV))
        print("   mpermH    [-] : ", _strvar(mpermH))
        print("   mpermV    [-] : ", _strvar(mpermV))

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

    # Print fullspace info
    if verb > 1 and isfullspace:
        print("\n>  MODEL IS A FULLSPACE; returning analytical " +
              "frequency-domain solution")

    return depth, res, aniso, epermH, epermV, mpermH, mpermV, isfullspace


def check_pole(inp, name, verb, intpts=-1):
    """Check dipole parameters.

    This check-function is called from one of the modelling routines in
    :mod:`model`.  Consult these modelling routines for a detailed description
    of the input parameters.

    Parameters
    ----------
    inp : list of floats or arrays
        Pole coordinates (m): [pole-x, pole-y, pole-z].

    name : str, {'src', 'rec'}
        Pole-type.

    verb : {0, 1, 2}
        Level of verbosity.

    intpts : int
        Number of integration points for bipole.

    Returns
    -------
    out : list
        List of pole coordinates [x, y, z].

    outbp : {tuple, None}
        - If pole is a dipole, None.
        - If pole is a bipole, (theta, phi, g_w):
            - theta : Horizontal pole angle.
            - phi : Vertical pole angle.
            - g_w : Integration weights.

    """

    # Flags depending on dipole/bipole
    isbipole = intpts >= 0
    if isbipole:
        narr = 6
    else:
        narr = 3

    # Check inp
    _check_shape(np.squeeze(inp), name, (narr,))
    inp[0] = _check_var(inp[0], float, 1, name+'-x')

    if not isbipole:  # z-value must be only one value if dipole
        inp[1] = _check_var(inp[1], float, 1, name+'-y', inp[0].shape)
        inp[2] = _check_var(inp[2], float, 1, name+'-z', (1,))
        out = inp     # if dipole, return verified input
        outbp = None  # if dipole, no outbp

    if isbipole:
        # Check inp
        inp[1] = _check_var(inp[1], float, 1, name+'-x1', inp[0].shape)
        inp[2] = _check_var(inp[2], float, 1, name+'-y', inp[0].shape)
        inp[3] = _check_var(inp[3], float, 1, name+'-y1', inp[0].shape)
        inp[4] = _check_var(inp[4], float, 1, name+'-z', inp[0].shape)
        inp[5] = _check_var(inp[5], float, 1, name+'-z1', inp[0].shape)

        # Get lengths in x/y/z-direction
        dx = np.squeeze(inp[1] - inp[0])
        dy = np.squeeze(inp[3] - inp[2])
        dz = np.squeeze(inp[5] - inp[4])

        # Check if inp is a dipole
        # (This is a problem, as we would could not define the angles then.)
        if dx == dy == dz == 0:
            print("* ERROR   :: <"+name+"> is a point dipole, use `dipole` " +
                  "instead of `bipole`/`srcbipole`.")
            raise ValueError('Bipole: dipole-'+name)

        # Get inp length and angles
        r = np.linalg.norm([dx, dy, dz])  # length of inp
        theta = np.arctan2(dy, dx)        # horizontal deviation from x-axis
        phi = np.pi/2-np.arccos(dz/r)     # vertical dev. from xy-plane down

        # Gauss quadrature, if intpts > 2; else set to center of inp
        intpts = _check_var(intpts, int, 0, 'intpts', ())

        if intpts > 2:  # Calculate the dipole positions
            # Get integration positions and weights
            g_x, g_w = special.p_roots(intpts)
            g_x *= r/2.0  # Adjust to inp length
            g_w /= 2.0    # Adjust to inp length (r/2), normalize (1/r)

            # Coordinate system is left-handed, positive z down
            # (Est-North-Depth).
            xinp = inp[0] + dx/2 + g_x*np.cos(phi)*np.cos(theta)
            yinp = inp[2] + dy/2 + g_x*np.cos(phi)*np.sin(theta)
            zinp = inp[4] + dz/2 + g_x*np.sin(phi)

        else:  # If intpts < 3: Calculate bipole at inp-centre for phi/theta
            intpts = 0
            xinp = np.array(inp[0] + dx/2)
            yinp = np.array(inp[2] + dy/2)
            zinp = np.array(inp[4] + dz/2)
            g_w = np.array([1])/r  # normalize for bipole length

        # Collect output list
        out = [xinp, yinp, zinp]
        outbp = (theta, phi, g_w)

    # Print spatial parameters
    if verb > 1:
        # Pole-type: src or rec
        if name == 'src':
            longname = '   Source(s)     : '
        else:
            longname = '   Receiver(s)   : '

        if isbipole:
            print(longname, str(inp[0].size), 'bipole(s)')
            print("     > intpts    : ", intpts)
            print("     > theta [°] : ", np.rad2deg(theta))
            print("     > phi   [°] : ", np.rad2deg(phi))
            print("     > length[m] : ", r)
            print("     > x_c   [m] : ", _strvar(inp[0][0] + dx/2))
            print("     > y_c   [m] : ", _strvar(inp[2][0] + dy/2))
            print("     > z_c   [m] : ", _strvar(inp[4][0] + dz/2))
        else:
            print(longname, str(inp[0].size), 'dipole(s)')
            tname = ['x  ', 'y  ', 'z  ']
            for i in range(3):
                if inp[i].size > 1:
                    print("     > "+tname[i]+"   [m] : ", str(inp[i].min()),
                          "-", str(inp[i].max()), " [min - max]")
                    if verb > 2:
                        print("                 : ", _strvar(inp[i]))
                else:
                    print("     > "+tname[i]+"   [m] : ", _strvar(inp[i]))

    return out, outbp


def check_srcrec(inp, name):
    """Check di-/bipole parameters.

    This check-function is called from one of the modelling routines in
    :mod:`model`.  Consult these modelling routines for a detailed description
    of the input parameters.

    Parameters
    ----------
    inp : list of floats or arrays
        Coordinates of inp (m):
        [dipole-x, dipole-y, dipole-z] or.
        [bipole-x0, bipole-x1, bipole-y0, bipole-y1, bipole-z0, bipole-z1].

    name : str, {'src', 'rec'}
        Pole-type.

    Returns
    -------
    inp : list
        As input, checked for type and length.

    ninp : int
        Number of sources/receivers

    ninpz : int
        Number of sources/receivers depths (ninpz is either 1 or ninp).

    isdipole : bool
        True if inp is a dipole.

    """

    def check_dipole(inp, name):
        """Check input for size and type."""
        # Check x
        inp[0] = _check_var(inp[0], float, 1, name+'-x')

        # Check y and ensure it has same dimension as x
        inp[1] = _check_var(inp[1], float, 1, name+'-y', inp[0].shape)

        # Check z
        inp[2] = _check_var(inp[2], float, 1, name+'-z')

        # Check if z is one value or has dimension of x
        zshape = inp[2].shape
        if zshape == (1,):
            pass
        elif zshape == inp[0].shape:
            # Check if all depths are the same, if so replace by one value
            if np.all(np.isclose(inp[2]-inp[2][0], 0)):
                inp[2] = np.array([inp[2][0]])
        else:
            print('* ERROR   :: Parameter ' + name + '-z has wrong shape! : ' +
                str(zshape) + ' instead of ' + str(inp[0].shape) + ' or (1,).')
            raise ValueError(name+'-z')

        return inp

    # Check length of inp.
    narr = len(inp)
    if narr not in [3, 6]:
        print('* ERROR   :: Parameter ' + name + ' has wrong length! : ' +
            str(narr) + ' instead of 3 (dipole) or 6 (bipole).')
        raise ValueError(name)

    # Flag if it is a dipole or not
    isdipole = narr == 3

    if isdipole:
        inp = check_dipole(inp, name)

    else:
        inp0 = check_dipole(inp[::2], name+'-1')
        inp1 = check_dipole(inp[1::2], name+'-2')

        # If one pole has a single depth, but the other has various
        # depths, we have to repeat the single depth, as we will have
        # to loop over them.
        if len(inp0[2]) != len(inp1[2]):
            if len(inp0[2]) == 1:
                inp0[2] = np.repeat(inp0[2], len(inp1[2]))
            else:
                inp1[2] = np.repeat(inp1[2], len(inp0[2]))

        # Collect elements
        inp = [inp0[0], inp1[0], inp0[1], inp1[1], inp0[2], inp1[2]]

    return inp, len(inp[0]), len(inp[2]), isdipole


def get_coords(src, rec, verb, intpts=(-1, -1)):
    """Get depths, offsets, angles, hence spatial input parameters.

    This check-function is called from one of the modelling routines in
    :mod:`model`.  Consult these modelling routines for a detailed description
    of the input parameters.

    Parameters
    ----------
    src : list of floats or arrays
        Source coordinates (m):
            - dipole: [src-x, src-y, src-z]
            - bipole: [src-x0, src-x1, src-y0, src-y1, src-z0, src-z1]

    rec : list of floats or arrays
        Receiver coordinates (m):
            - dipole: [src-x, src-y, src-z]
            - bipole: [src-x0, src-x1, src-y0, src-y1, src-z0, src-z1]

    verb : {0, 1, 2}
        Level of verbosity.

    intpts : tuple (int, int)
        Number of integration points for bipole for (src, rec).
            - nr < 0      : dipole
            - 0 <= nr < 3 : bipole, but calculated as dipole at centerpoint
            - nr >= 3     : bipole


    Returns
    -------
    zsrc : array of float
        Depth(s) of src (plural only if bipole).

    zrec : array of float
        Depth(s) of rec (plural only if bipole).

    off : array of floats
        Offsets

    angle : array of floats
        Angles

    nsrc: int
        Number of bipole sources

    nrec: int
        Number of receivers

    srcrecbp: tuple
        (srcbp, recbp)
        If src/rec is dipole: None
        If src/rec is bipole: tuple containing (theta, phi, g_w)

    """

    # Check source(s)
    src, srcbp = check_pole(src, 'src', verb, intpts[0])
    nsrc = src[0].size

    # Check receiver(s)
    rec, recbp = check_pole(rec, 'rec', verb, intpts[1])
    nrec = rec[0].size

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
    # src-rec configurations that have the same offset and angle.

    # Minimum offset to avoid singularities at off = 0 m.
    # => min_off is defined at the start of this file
    ioff = np.where(off < min_off)
    off[ioff] = min_off
    angle[ioff] = np.nan
    if np.size(ioff) != 0 and verb > 0:
        print('* WARNING :: Offsets <', min_off, 'm are set to', min_off, 'm!')

    return src[2], rec[2], off, angle, nsrc, nrec, (srcbp, recbp)


def get_coords_tmp(src, rec, nsrc, nrec, verb):
    """Get depths, offsets, angles, hence spatial input parameters.

    This check-function is called from one of the modelling routines in
    :mod:`model`.  Consult these modelling routines for a detailed description
    of the input parameters.

    Parameters
    ----------
    src : list of floats or arrays
        Source coordinates (m):
            - dipole: [src-x, src-y, src-z]
            - bipole: [src-x0, src-x1, src-y0, src-y1, src-z0, src-z1]

    rec : list of floats or arrays
        Receiver coordinates (m):
            - dipole: [src-x, src-y, src-z]
            - bipole: [src-x0, src-x1, src-y0, src-y1, src-z0, src-z1]

    verb : {0, 1, 2}
        Level of verbosity.

    intpts : tuple (int, int)
        Number of integration points for bipole for (src, rec).
            - nr < 0      : dipole
            - 0 <= nr < 3 : bipole, but calculated as dipole at centerpoint
            - nr >= 3     : bipole


    Returns
    -------
    zsrc : array of float
        Depth(s) of src (plural only if bipole).

    zrec : array of float
        Depth(s) of rec (plural only if bipole).

    off : array of floats
        Offsets

    angle : array of floats
        Angles

    nsrc: int
        Number of bipole sources

    nrec: int
        Number of receivers

    srcrecbp: tuple
        (srcbp, recbp)
        If src/rec is dipole: None
        If src/rec is bipole: tuple containing (theta, phi, g_w)

    """

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
    # src-rec configurations that have the same offset and angle.

    # Minimum offset to avoid singularities at off = 0 m.
    # => min_off is defined at the start of this file
    ioff = np.where(off < min_off)
    off[ioff] = min_off
    angle[ioff] = np.nan
    if np.size(ioff) != 0 and verb > 0:
        print('* WARNING :: Offsets <', min_off, 'm are set to', min_off, 'm!')

    return src[2], rec[2], off, angle


def get_thetaphi(inp, iz, nrinpz, intpts, isdipole, name, verb):
    """TODO"""

    # Get this bipole
    if nrinpz == 1:
        tinp = inp
    else:
        if isdipole:
            tinp = [inp[0][iz], inp[1][iz], inp[2][iz]]
        else:
            tinp = [inp[0][iz], inp[1][iz], inp[2][iz],
                    inp[3][iz], inp[4][iz], inp[5][iz]]

    # Get number of integration points and angles for source
    if isdipole:
        intpts = 1
        theta = None
        phi = None
        g_w = np.array([1])
        tout = tinp
    else:
        # Get lengths in each direction
        dx = np.squeeze(tinp[1] - tinp[0])
        dy = np.squeeze(tinp[3] - tinp[2])
        dz = np.squeeze(tinp[5] - tinp[4])

        # Check if tinp is a dipole
        # (This is a problem, as we would could not define the angles then.)
        if np.all(dx == 0) and np.all(dy == 0) and np.all(dz == 0):
            print("* ERROR   :: <"+name+"> is a point dipole, use `dipole` " +
                    "instead of `bipole`/`srcbipole`.")
            raise ValueError('Bipole: dipole-'+name)

        # Get bipole length length and angles
        dl = np.linalg.norm([dx, dy, dz], axis=0)  # length of tinp-bipole
        theta = np.arctan2(dy, dx)      # horizontal deviation from x-axis
        phi = np.pi/2-np.arccos(dz/dl)  # vertical deviation from xy-plane down

        # Gauss quadrature, if intpts > 2; else set to center of tinp
        intpts = _check_var(intpts, int, 0, 'intpts', ())
        if intpts > 2:  # Calculate the dipole positions
            # Get integration positions and weights
            g_x, g_w = special.p_roots(intpts)
            g_x = np.outer(g_x, dl/2.0)  # Adjust to tinp length
            g_w /= 2.0    # Adjust to tinp length (dl/2), normalize (1/dl)

            # Coordinate system is left-handed, positive z down
            # (Est-North-Depth).
            xinp = tinp[0] + dx/2 + g_x*np.cos(phi)*np.cos(theta)
            yinp = tinp[2] + dy/2 + g_x*np.cos(phi)*np.sin(theta)
            zinp = tinp[4] + dz/2 + g_x*np.sin(phi)

        else:  # If intpts < 3: Calculate bipole at tinp-centre for phi/theta
            intpts = 1
            xinp = np.array(tinp[0] + dx/2)
            yinp = np.array(tinp[2] + dy/2)
            zinp = np.array(tinp[4] + dz/2)
            g_w = np.array([1])/dl  # normalize for bipole length

        # Collect output list; rounding coordinates to same precision as min_off
        rndco = int(np.round(np.log10(1/min_off)))
        tout = [np.round(xinp, rndco), np.round(yinp, rndco), np.round(zinp,
               rndco)]

    # Print spatial parameters
    if verb > 1:
        # Pole-type: src or rec
        if name == 'src':
            longname = '   Source(s)     : '
        else:
            longname = '   Receiver(s)   : '

        if isdipole:
            print(longname, str(tout[0].size), 'dipole(s)')
            tname = ['x  ', 'y  ', 'z  ']
            for i in range(3):
                if tout[i].size > 1:
                    print("     > "+tname[i]+"   [m] : ", str(tout[i].min()),
                          "-", str(tout[i].max()), " [min - max]")
                    if verb > 2:
                        print("                 : ", _strvar(tout[i]))
                else:
                    print("     > "+tname[i]+"   [m] : ", _strvar(tout[i]))
        else:
            print(longname, str(tout[0].size), 'bipole(s)')
            if intpts < 3:
                print("     > intpts    :  1 (as dipole)")
            else:
                print("     > intpts    : ", intpts)
            print("     > theta [°] : ", np.rad2deg(theta))
            print("     > phi   [°] : ", np.rad2deg(phi))
            print("     > length[m] : ", dl)
            print("     > x_c   [m] : ", _strvar(tinp[0][0] + dx/2))
            print("     > y_c   [m] : ", _strvar(tinp[2][0] + dy/2))
            print("     > z_c   [m] : ", _strvar(tinp[4][0] + dz/2))

    return tout, theta, phi, g_w, intpts


def check_depth(zsrc, zrec, depth):
    """Check layer in which source/receiver reside.

    Note: If zsrc or zrec are on a layer interface, the layer above the
          interface is chosen.

    This check-function is called from one of the modelling routines in
    :mod:`model`.  Consult these modelling routines for a detailed description
    of the input parameters.

    Parameters
    ----------
    zsrc : array of float
        Depth(s) of src (plural only if bipole).

    zrec : array of float
        Depth(s) of rec (plural only if bipole).

    depth : array
        Depths of layer interfaces.


    Returns
    -------
    lsrc : int or array_like of int
        Layer number(s) in which src resides (plural only if bipole).

    lrec : int or array_like of int
        Layer number(s) in which rec resides (plural only if bipole).

    """

    #  depth = [-infty : last interface]; create additional depth-array
    # pdepth = [fist interface : +infty]
    pdepth = np.concatenate((depth[1:], np.array([np.infty])))

    # Broadcast arrays
    b_depth = depth[None, :]
    b_pdepth = pdepth[None, :]
    b_zsrc = zsrc[:, None]
    b_zrec = zrec[:, None]

    # Get layers
    lsrc = np.where((b_depth < b_zsrc)*(b_pdepth >= b_zsrc))[1]
    lrec = np.where((b_depth < b_zrec)*(b_pdepth >= b_zrec))[1]

    # Return; squeeze in case of only one src/rec-depth
    return np.squeeze(lsrc), np.squeeze(lrec)


def check_hankel(ht, htarg, verb):
    """Check Hankel transform parameters.

    This check-function is called from one of the modelling routines in
    :mod:`model`.  Consult these modelling routines for a detailed description
    of the input parameters.

    Parameters
    ----------
    ht : {'fht', 'qwe'}
        Flag to choose the Hankel transform.

    htarg : str or filter from empymod.filters or array_like,
        Depends on the value for `ht`.

    verb : {0, 1, 2}
        Level of verbosity.


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
            print("   Hankel        :  Fast Hankel Transform")
            print("     > Filter    :  " + fhtfilt.name)

    elif ht in ['qwe', 'hqwe']:
        # Rename ht
        ht = 'hqwe'

        # Get and check input or set defaults
        if not htarg:
            htarg = []

        # rtol : 1e-12 is low for accurate results
        try:
            rtol = _check_var(htarg[0], float, 0, 'qwe: rtol', ())
        except:
            rtol = np.array(1e-12, dtype=float)

        # atol : 1e-30 is low for accurate results
        try:
            atol = _check_var(htarg[1], float, 0, 'qwe: atol', ())
        except:
            atol = np.array(1e-30, dtype=float)

        # nquad : 51 is relatively high
        try:
            nquad = _check_var(htarg[2], int, 0, 'qwe: nquad', ())
        except:
            nquad = np.array(51, dtype=int)

        # maxint :  40/100 is relatively high
        #             40 : 11-15, 21-25, 33-35, 41-45, 51-55
        #            100 : 16/26, 31/32, 46/56, 61-66
        try:
            maxint = _check_var(htarg[3], int, 0, 'qwe: maxint', ())
        except:
            maxint = np.array(100, dtype=int)

        # pts_per_dec : 80 is relatively high
        try:
            pts_per_dec = _check_var(htarg[4], int, 0, 'qwe: pts_per_dec', ())
        except:
            pts_per_dec = np.array(80, dtype=int)

        # Assemble htarg
        htarg = (rtol, atol, nquad, maxint, pts_per_dec)

        # If verbose, print Hankel transform information
        if verb > 1:
            print("   Hankel        :  Quadrature-with-Extrapolation")
            print("     > rtol      :  " + str(htarg[0]))
            print("     > atol      :  " + str(htarg[1]))
            print("     > nquad     :  " + str(htarg[2]))
            print("     > maxint    :  " + str(htarg[3]))

    else:
        print("* ERROR   :: <ht> must be one of: ['fht', 'qwe'];" +
              " <ht> provided: " + str(ht))
        raise ValueError('ht')

    return ht, htarg


def check_frequency(freq, res, aniso, epermH, epermV, mpermH, mpermV, verb):
    """Calculate frequency-dependent parameters.

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

    epermH : array_like
        Horizontal electric permittivities epsilon_h (-); #epermH = #res.

    epermV : array_like
        Vertical electric permittivities epsilon_v (-); #epermV = #res.

    mpermH : array_like
        Horizontal magnetic permeabilities mu_h (-); #mpermH = #res.

    mpermV : array_like
        Vertical magnetic permeabilities mu_v (-); #mpermV = #res.

    verb : {0, 1, 2}
        Level of verbosity.


    Returns
    -------
    freq : float
        Frequency, checked for size and assured min_freq.

    etaH : array
        Parameters etaH, same size as provided resistivity.

    etaV : array
        Parameters etaV, same size as provided resistivity.

    zetaH : array
        Parameters zetaH, same size as provided resistivity.

    zetaV : array
        Parameters zetaV, same size as provided resistivity.

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
        print("   freq     [Hz] : ", str(freq.min()), "-", str(freq.max()),
              ";", str(freq.size), " [min-max; #]")
        if verb > 2:
            print("                 : ", _strvar(freq))

    # Calculate eta and zeta (horizontal and vertical)
    etaH = 1/res + np.outer(2j*np.pi*freq, epermH*epsilon_0)
    etaV = 1/(res*aniso*aniso) + np.outer(2j*np.pi*freq, epermV*epsilon_0)
    zetaH = np.outer(2j*np.pi*freq, mpermH*mu_0)
    zetaV = np.outer(2j*np.pi*freq, mpermV*mu_0)

    return freq, etaH, etaV, zetaH, zetaV


def check_opt(opt, loop, ht, htarg, verb):
    """Check optimization parameters.

    This check-function is called from one of the modelling routines in
    :mod:`model`.  Consult these modelling routines for a detailed description
    of the input parameters.

    Parameters
    ----------
    opt : {None, 'parallel', 'spline'}
        Optimization flag.

    loop : {None, 'freq', 'off'}
        Loop flag.

    ht : str
        Flag to choose the Hankel transform.

    htarg : array_like,
        Depends on the value for `ht`.

    verb : {0, 1, 2}
        Level of verbosity.


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
            print("   Hankel Opt.   :  Use spline")
            pstr = "     > pts/dec   :  "
            if ht == 'hqwe':
                print(pstr + str(htarg[4]))
            else:
                if htarg[1]:
                    print(pstr + str(htarg[1]))
                else:
                    print(pstr + 'Defined by filter (lagged)')
        elif use_ne_eval:
            print("   Hankel Opt.   :  Use parallel")
        else:
            print("   Hankel Opt.   :  None")

        if loop_off:
            print("   Loop over     :  Offsets")
        elif loop_freq:
            print("   Loop over     :  Frequencies")
        else:
            print("   Loop over     :  None (all vectorized)")

    return use_spline, use_ne_eval, loop_freq, loop_off


def check_time(freqtime, signal, ft, ftarg, verb):
    """Check time domain specific input parameters.

    This check-function is called from one of the modelling routines in
    :mod:`model`.  Consult these modelling routines for a detailed description
    of the input parameters.

    Parameters
    ----------
    freqtime : array_like
        Frequencies f (Hz) if `signal` == None, else times t (s).

    signal : {None, 0, 1, -1}
        Source signal:
            - None: Frequency-domain response
            - -1 : Switch-off time-domain response
            - 0 : Impulse time-domain response
            - +1 : Switch-on time-domain response

    ft : {'sin', 'cos', 'qwe', 'fftlog'}
        Flag for Fourier transform, only used if `signal` != None.

    ftarg : str or filter from empymod.filters or array_like,
        Only used if `signal` !=None. Depends on the value for `ft`:

    verb : {0, 1, 2}
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

    if signal is None:
        return None, freqtime, ft, ftarg
    elif int(signal) not in [-1, 0, 1]:
        print("* ERROR   :: <signal> must be one of: [None, -1, 0, 1]; " +
              "<signal> provided: "+str(signal))
        raise ValueError('signal')

    # Check time
    time = _check_var(freqtime, float, 1, 'time')

    # Minimum time to avoid division by zero  at time = 0 s.
    # => min_time is defined at the start of this file
    itime = np.where(time < min_time)
    time[itime] = min_time
    if verb > 0 and np.size(itime) != 0:
        print('* WARNING :: Times <', min_time, 's are set to', min_time, 's!')
    if verb > 1:
        print("   time      [s] : ", str(time.min()), "-", str(time.max()),
              ";", str(time.size), " [min-max; #]")
        if verb > 2:
            print("                 : ", _strvar(time))

    # Ensure ft is all lowercase
    ft = ft.lower()

    if ft in ['cos', 'sin', 'fft']:  # If Cosine/Sine, check filter setting

        # If `ft='fft'`, we assume that it run the check before, and get
        # sin/cos from ftarg. If not, defaults to 'sin'. To ensure that this
        # check can be re-run without failing.
        if ft == 'fft':
            try:
                ft = ftarg[2]
            except:
                ft = 'sin'

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
        ftarg = (fftfilt, pts_per_dec, ft)

        # If verbose, print Fourier transform information
        if verb > 1:
            if ft == 'sin':
                print("   Fourier       :  Sine-Filter")
            else:
                print("   Fourier       :  Cosine-Filter")
            print("     > Filter    :  " + ftarg[0].name)
            pstr = "     > pts/dec   :  "
            if ftarg[1]:
                print(pstr + str(ftarg[1]))
            else:
                print(pstr + 'Defined by filter (lagged)')

        # Get required frequencies
        # (multiply time by 2Pi, as calculation is done in angular frequencies)
        freq, _ = transform.get_spline_values(ftarg[0], 2*np.pi*time, ftarg[1])
        freq = np.squeeze(freq)

        # Rename ft
        ft = 'fft'

    elif ft in ['qwe', 'fqwe']:    # QWE
        # Rename ft
        ft = 'fqwe'

        # Get and check input or set defaults
        if not ftarg:  # Default values
            ftarg = []

        try:  # rtol
            rtol = _check_var(ftarg[0], float, 0, 'qwe: rtol', ())
        except:
            rtol = np.array(1e-8, dtype=float)

        try:  # atol
            atol = _check_var(ftarg[1], float, 0, 'qwe: atol', ())
        except:
            atol = np.array(1e-20, dtype=float)

        try:  # nquad
            nquad = _check_var(ftarg[2], int, 0, 'qwe: nquad', ())
        except:
            nquad = np.array(21, dtype=int)

        try:  # maxint
            maxint = _check_var(ftarg[3], int, 0, 'qwe: maxint', ())
        except:
            maxint = np.array(200, dtype=int)

        try:  # pts_per_dec
            pts_per_dec = _check_var(ftarg[4], int, 0, 'qwe: pts_per_dec', ())
        except:
            pts_per_dec = np.array(20, dtype=int)

        # Assemble ftarg
        ftarg = (rtol, atol, nquad, maxint, pts_per_dec)

        # If verbose, print Fourier transform information
        if verb > 1:
            print("   Fourier        :  Quadrature-with-Extrapolation")
            print("     > rtol      :  " + str(ftarg[0]))
            print("     > atol      :  " + str(ftarg[1]))
            print("     > nquad     :  " + str(ftarg[2]))
            print("     > maxint    :  " + str(ftarg[3]))
            print("     > pts/dec   :  " + str(ftarg[4]))

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
            pts_per_dec = np.array(10, dtype=int)

        try:  # add_dec
            add_dec = _check_var(ftarg[1], float, 1, 'fftlog: add_dec', (2,))
        except:
            add_dec = np.array([-2, 1], dtype=float)

        try:  # q
            q = _check_var(ftarg[2], float, 0, 'fftlog: q', ())
            # Restrict q to +/- 1
            if np.abs(q) > 1:
                q = np.sign(q)
        except:
            q = np.array(0, dtype=float)

        # If verbose, print Fourier transform information
        if verb > 1:
            print("   Fourier        :  FFTLog ")
            print("     > pts/dec    :  " + str(pts_per_dec))
            print("     > add_dec    :  " + str(add_dec))
            print("     > q          :  " + str(q))

        # Calculate minimum and maximum required frequency
        minf = np.log10(1/time.max()) + add_dec[0]
        maxf = np.log10(1/time.min()) + add_dec[1]
        n = np.int(maxf - minf)*pts_per_dec

        # Initialize FFTLog, get required parameters
        freq, tcalc, dlnr, kr, rk = transform.fhti(minf, maxf, n, q)

        # Assemble ftarg
        # Keep first 3 entries, so re-running this check is stable
        ftarg = (pts_per_dec, add_dec, q, tcalc, dlnr, kr, rk, q)

    else:
        print("* ERROR   :: <ft> must be one of: ['cos', 'sin', 'qwe', " +
              "'fftlog']; <ft> provided: "+str(ft))
        raise ValueError('ft')

    return time, freq, ft, ftarg


# 3. Internal utilities

def _strvar(a, prec='{:G}'):
    """Return variable as a string to print, with given precision."""
    return ' '.join([prec.format(i) for i in np.atleast_1d(a)])


def _check_var(var, dtype, ndmin, name, shape=None):
    """Return variable as array of dtype, ndmin; shape-checked."""
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


def printstartfinish(verb, inp=None, kcount=0):
    """Print start and finish with time measure."""
    if inp:
        print('\n:: empymod END; runtime = ' +
              str(timedelta(seconds=default_timer() - inp)) + ' :: ' +
              str(kcount) + ' kernel call(s)\n')
    else:
        t0 = default_timer()
        if verb > 1:
            print("\n:: empymod START  ::\n\n>  INPUT CHECK")
        return t0
