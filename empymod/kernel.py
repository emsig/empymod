r"""

:mod:`kernel` -- Kernel calculation
===================================

Kernel of ``empymod``, calculates the wavenumber-domain electromagnetic
response. Plus analytical full- and half-space solutions.

The functions ``wavenumber``, ``angle_factor``, ``fullspace``, ``greenfct``,
``reflections``, and ``fields`` are based on source files (specified in each
function) from the source code distributed with [HuTS15]_, which can be found
at `software.seg.org/2015/0001 <https://software.seg.org/2015/0001>`_.  These
functions are (c) 2015 by Hunziker et al. and the Society of Exploration
Geophysicists, https://software.seg.org/disclaimer.txt.  Please read the
NOTICE-file in the root directory for more information regarding the involved
licenses.

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


import numpy as np
from scipy import special  # Only used for halfspace solution

np.seterr(all='ignore')

__all__ = ['wavenumber', 'angle_factor', 'fullspace', 'greenfct',
           'reflections', 'fields', 'halfspace']


# Wavenumber-frequency domain kernel

def wavenumber(zsrc, zrec, lsrc, lrec, depth, etaH, etaV, zetaH, zetaV, lambd,
               ab, xdirect, msrc, mrec, use_ne_eval):
    r"""Calculate wavenumber domain solution.

    Return the wavenumber domain solutions ``PJ0``, ``PJ1``, and ``PJ0b``,
    which have to be transformed with a Hankel transform to the frequency
    domain. ``PJ0``/``PJ0b`` and ``PJ1`` have to be transformed with Bessel
    functions of order 0 (:math:`J_0`) and 1 (:math:`J_1`), respectively.

    This function corresponds loosely to equations 105--107, 111--116,
    119--121, and 123--128 in [HuTS15]_, and equally loosely to the file
    ``kxwmod.c``.

    [HuTS15]_ uses Bessel functions of orders 0, 1, and 2 (:math:`J_0, J_1,
    J_2`). The implementations of the *Fast Hankel Transform* and the
    *Quadrature-with-Extrapolation* in ``transform`` are set-up with Bessel
    functions of order 0 and 1 only. This is achieved by applying the
    recurrence formula

    .. math:: J_2(kr) = \frac{2}{kr} J_1(kr) - J_0(kr) \ .


    .. note::

        ``PJ0`` and ``PJ0b`` could theoretically be added here into one, and
        then be transformed in one go.  However, ``PJ0b`` has to be multiplied
        by ``factAng`` later. This has to be done after the Hankel transform
        for methods which make use of spline interpolation, in order to work
        for offsets that are not in line with each other.

    This function is called from one of the Hankel functions in
    :mod:`transform`.  Consult the modelling routines in :mod:`model` for a
    description of the input and output parameters.

    If you are solely interested in the wavenumber-domain solution you can call
    this function directly. However, you have to make sure all input arguments
    are correct, as no checks are carried out here.

    """

    # ** CALCULATE GREEN'S FUNCTIONS
    # Shape of PTM, PTE: (nfreq, noffs, nfilt)
    PTM, PTE = greenfct(zsrc, zrec, lsrc, lrec, depth, etaH, etaV, zetaH,
                        zetaV, lambd, ab, xdirect, msrc, mrec, use_ne_eval)

    # ** AB-SPECIFIC COLLECTION OF PJ0, PJ1, AND PJ0b

    # Pre-allocate output
    PJ0 = None
    PJ1 = None
    PJ0b = None

    # Calculate Ptot which is used in all cases
    Ptot = (PTM + PTE)/(4*np.pi)

    # If rec is magnetic switch sign (reciprocity MM/ME => EE/EM).
    if mrec:
        sign = -1
    else:
        sign = 1

    # Group into PJ0 and PJ1 for J0/J1 Hankel Transform
    if ab in [11, 12, 21, 22, 14, 24, 15, 25]:    # Eqs 105, 106, 111, 112,
        # J2(kr) = 2/(kr)*J1(kr) - J0(kr)         #     119, 120, 123, 124
        if ab in [14, 22]:
            sign *= -1
        PJ0b = sign/2*Ptot*lambd
        PJ1 = -sign*Ptot
        if ab in [11, 22, 24, 15]:
            if ab in [22, 24]:
                sign *= -1
            PJ0 = sign*(PTM - PTE)/(8*np.pi)*lambd

    elif ab in [13, 23, 31, 32, 34, 35, 16, 26]:  # Eqs 107, 113, 114, 115,
        PJ1 = sign*Ptot*lambd*lambd               # .   121, 125, 126, 127
        if ab in [34, 26]:
            PJ1 *= -1

    elif ab in [33, ]:                            # Eq 116
        PJ0 = sign*Ptot*lambd*lambd*lambd

    # Return PJ0, PJ1, PJ0b
    return PJ0, PJ1, PJ0b


def greenfct(zsrc, zrec, lsrc, lrec, depth, etaH, etaV, zetaH, zetaV, lambd,
             ab, xdirect, msrc, mrec, use_ne_eval):
    r"""Calculate Green's function for TM and TE.

    .. math:: \tilde{g}^{tm}_{hh}, \tilde{g}^{tm}_{hz},
              \tilde{g}^{tm}_{zh}, \tilde{g}^{tm}_{zz},
              \tilde{g}^{te}_{hh}, \tilde{g}^{te}_{zz}

    This function corresponds to equations 108--110, 117/118, 122; 89--94,
    A18--A23, B13--B15; 97--102 A26--A31, and B16--B18 in [HuTS15]_, and
    loosely to the corresponding files ``Gamma.F90``, ``Wprop.F90``,
    ``Ptotalx.F90``, ``Ptotalxm.F90``, ``Ptotaly.F90``, ``Ptotalym.F90``,
    ``Ptotalz.F90``, and ``Ptotalzm.F90``.

    The Green's functions are multiplied according to Eqs 105-107, 111-116,
    119-121, 123-128; with the factors inside the integrals.

    This function is called from the function :mod:`kernel.wavenumber`.

    """
    # GTM/GTE have shape (frequency, offset, lambda).
    # gamTM/gamTE have shape (frequency, offset, layer, lambda):

    # Reciprocity switches for magnetic receivers
    if mrec:
        if msrc:  # If src is also magnetic, switch eta and zeta (MM => EE).
            # G^mm_ab(s, r, e, z) = -G^ee_ab(s, r, -z, -e)
            etaH, zetaH = -zetaH, -etaH
            etaV, zetaV = -zetaV, -etaV
        else:  # If src is electric, swap src and rec (ME => EM).
            # G^me_ab(s, r, e, z) = -G^em_ba(r, s, e, z)
            zsrc, zrec = zrec, zsrc
            lsrc, lrec = lrec, lsrc

    for TM in [True, False]:

        # Continue if Green's function not required
        if TM and ab in [16, 26]:
            continue
        elif not TM and ab in [13, 23, 31, 32, 33, 34, 35]:
            continue

        # Define eta/zeta depending if TM or TE
        if TM:
            e_zH, e_zV, z_eH = etaH, etaV, zetaH   # TM: zetaV not used
        else:
            e_zH, e_zV, z_eH = zetaH, zetaV, etaH  # TE: etaV not used

        # Uppercase gamma
        if use_ne_eval:
            ez_ratio = (e_zH/e_zV)[:, None, :, None]  # NOQA
            ez_prod = (z_eH*e_zH)[:, None, :, None]  # NOQA
            lambd2 = use_ne_eval("lambd*lambd")[None, :, None, :]  # NOQA
            Gam = use_ne_eval("sqrt(ez_ratio*lambd2 + ez_prod)")
        else:
            Gam = np.sqrt((e_zH/e_zV)[:, None, :, None] *
                          (lambd*lambd)[None, :, None, :] +
                          (z_eH*e_zH)[:, None, :, None])

        # Gamma in receiver layer
        lrecGam = Gam[:, :, lrec, :]

        # Reflection (coming from below (Rp) and above (Rm) rec)
        if depth.size > 1:  # Only if more than 1 layer
            Rp, Rm = reflections(depth, e_zH, Gam, lrec, lsrc, use_ne_eval)

            # Field propagators
            # (Up- (Wu) and downgoing (Wd), in rec layer); Eq 74
            if lrec != depth.size-1:  # No upgoing field prop. if rec in last
                ddepth = depth[lrec + 1] - zrec
                if use_ne_eval:
                    Wu = use_ne_eval("exp(-lrecGam*ddepth)")
                else:
                    Wu = np.exp(-lrecGam*ddepth)
            else:
                Wu = np.zeros_like(lrecGam)
            if lrec != 0:     # No downgoing field propagator if rec in first
                ddepth = zrec - depth[lrec]
                if use_ne_eval:
                    Wd = use_ne_eval("exp(-lrecGam*ddepth)")
                else:
                    Wd = np.exp(-lrecGam*ddepth)
            else:
                Wd = np.zeros_like(lrecGam)

            # Field at rec level (coming from below (Pu) and above (Pd) rec)
            Pu, Pd = fields(depth, Rp, Rm, Gam, lrec, lsrc, zsrc, ab, TM,
                            use_ne_eval)

        # Green's functions
        if lsrc == lrec:  # Rec in src layer; Eqs 108, 109, 110, 117, 118, 122

            # Green's function depending on <ab>
            if depth.size == 1:  # If only one layer, no reflections/fields
                green = np.zeros_like(lrecGam)
            elif ab in [13, 23, 31, 32, 14, 24, 15, 25]:
                green = Pu*Wu - Pd*Wd
            else:
                green = Pu*Wu + Pd*Wd

            # Direct field, if it is computed in the wavenumber domain
            if not xdirect:
                # Direct field
                directf = np.exp(-lrecGam*abs(zsrc - zrec))

                # Swap TM for certain <ab>
                if TM and ab in [11, 12, 13, 14, 15, 21, 22, 23, 24, 25]:
                    directf *= -1

                # Multiply by zrec-zsrc-sign for certain <ab>
                if ab in [13, 14, 15, 23, 24, 25, 31, 32]:
                    directf *= np.sign(zrec - zsrc)

                # Add direct field to Green's function
                green += directf

        else:

            # Calculate exponential factor
            if lrec == depth.size-1:
                ddepth = 0
            else:
                ddepth = depth[lrec+1] - depth[lrec]
            if use_ne_eval:
                fexp = use_ne_eval("exp(-lrecGam*ddepth)")
            else:
                fexp = np.exp(-lrecGam*ddepth)

            # Sign-switch for Green calculation
            if TM and ab in [11, 12, 13, 21, 22, 23, 14, 24, 15, 25]:
                pmw = -1
            else:
                pmw = 1

            if lrec < lsrc:  # Rec above src layer: Pd not used
                #              Eqs 89-94, A18-A23, B13-B15
                green = Pu*(Wu + pmw*Rm[:, :, 0, :]*fexp*Wd)

            elif lrec > lsrc:  # rec below src layer: Pu not used
                #                Eqs 97-102 A26-A30, B16-B18
                green = Pd*(pmw*Wd + Rp[:, :, abs(lsrc-lrec), :]*fexp*Wu)

        # Store in corresponding variable
        if TM:
            gamTM, GTM = Gam, green
        else:
            gamTE, GTE = Gam, green

    # ** AB-SPECIFIC FACTORS AND CALCULATION OF PTOT'S
    # These are the factors inside the integrals
    # Eqs 105-107, 111-116, 119-121, 123-128

    # PTM, PTE
    if ab in [11, 12, 21, 22]:
        PTM = GTM*gamTM[:, :, lrec, :]/etaH[:, None, lrec, None]
        PTE = zetaH[:, None, lsrc, None]*GTE/gamTE[:, :, lsrc, :]
    elif ab in [14, 15, 24, 25]:
        PTM = ((etaH[:, lsrc]/etaH[:, lrec])[:, None, None] *
               GTM*gamTM[:, :, lrec, :]/gamTM[:, :, lsrc, :])
        PTE = GTE
    elif ab in [13, 23]:
        PTM = -((etaH[:, lsrc]/etaH[:, lrec]/etaV[:, lsrc])[:, None, None] *
                GTM*gamTM[:, :, lrec, :]/gamTM[:, :, lsrc, :])
        PTE = 0
    elif ab in [31, 32]:
        PTM = GTM/etaV[:, None, lrec, None]
        PTE = 0
    elif ab in [34, 35]:
        PTM = ((etaH[:, lsrc]/etaV[:, lrec])[:, None, None] *
               GTM/gamTM[:, :, lsrc, :])
        PTE = 0
    elif ab in [16, 26]:
        PTM = 0
        PTE = ((zetaH[:, lsrc]/zetaV[:, lsrc])[:, None, None] *
               GTE/gamTE[:, :, lsrc, :])
    elif ab in [33, ]:
        PTM = ((etaH[:, lsrc]/etaV[:, lsrc]/etaV[:, lrec])[:, None, None] *
               GTM/gamTM[:, :, lsrc, :])
        PTE = 0

    # Return Green's functions
    return PTM, PTE


def reflections(depth, e_zH, Gam, lrec, lsrc, use_ne_eval):
    r"""Calculate Rp, Rm.

    .. math:: R^\pm_n, \bar{R}^\pm_n

    This function corresponds to equations 64/65 and A-11/A-12 in
    [HuTS15]_, and loosely to the corresponding files ``Rmin.F90`` and
    ``Rplus.F90``.

    This function is called from the function :mod:`kernel.greenfct`.

    """

    # Loop over Rp, Rm
    for plus in [True, False]:

        # Switches depending if plus or minus
        if plus:
            pm = 1
            layer_count = np.arange(depth.size-2, min(lrec, lsrc)-1, -1)
            izout = abs(lsrc-lrec)
            minmax = max(lrec, lsrc)
        else:
            pm = -1
            layer_count = np.arange(1, max(lrec, lsrc)+1, 1)
            izout = 0
            minmax = -min(lrec, lsrc)

        # If rec in last  and rec below src (plus) or
        # if rec in first and rec above src (minus), shift izout
        shiftplus = lrec < lsrc and lrec == 0 and not plus
        shiftminus = lrec > lsrc and lrec == depth.size-1 and plus
        if shiftplus or shiftminus:
            izout -= pm

        # Pre-allocate Ref
        Ref = np.zeros((Gam.shape[0], Gam.shape[1], abs(lsrc-lrec)+1,
                        Gam.shape[3]), dtype=Gam.dtype)

        # Calculate the reflection
        for iz in layer_count:

            # Eqs 65, A-12
            e_zHa = e_zH[:, None, iz+pm, None]
            Gama = Gam[:, :, iz, :]
            e_zHb = e_zH[:, None, iz, None]
            Gamb = Gam[:, :, iz+pm, :]
            if use_ne_eval:
                rlocstr = "(e_zHa*Gama - e_zHb*Gamb)/(e_zHa*Gama + e_zHb*Gamb)"
                rloc = use_ne_eval(rlocstr)
            else:
                rloca = e_zHa*Gama
                rlocb = e_zHb*Gamb
                rloc = (rloca - rlocb)/(rloca + rlocb)

            # In first layer tRef = rloc
            if iz == layer_count[0]:
                tRef = rloc.copy()
            else:
                ddepth = depth[iz+1+pm]-depth[iz+pm]

                # Eqs 64, A-11
                if use_ne_eval:
                    term = use_ne_eval("tRef*exp(-2*Gamb*ddepth)")
                    tRef = use_ne_eval("(rloc + term)/(1 + rloc*term)")
                else:
                    term = tRef*np.exp(-2*Gamb*ddepth)  # NOQA
                    tRef = (rloc + term)/(1 + rloc*term)

            # The global reflection coefficient is given back for all layers
            # between and including src- and rec-layer
            if lrec != lsrc and pm*iz <= minmax:
                Ref[:, :, izout, :] = tRef[:]
                izout -= pm

        # If lsrc = lrec, we just store the last values
        if lsrc == lrec and layer_count.size > 0:
            Ref = tRef

        # Store Ref in Rm/Rp
        if plus:
            Rm = Ref
        else:
            Rp = Ref

    # Return reflections (minus and plus)
    return Rm, Rp


def fields(depth, Rp, Rm, Gam, lrec, lsrc, zsrc, ab, TM, use_ne_eval):
    r"""Calculate Pu+, Pu-, Pd+, Pd-.

    .. math:: P^{u\pm}_s, P^{d\pm}_s, \bar{P}^{u\pm}_s, \bar{P}^{d\pm}_s;
          P^{u\pm}_{s-1}, P^{u\pm}_n, \bar{P}^{u\pm}_{s-1}, \bar{P}^{u\pm}_n;
          P^{d\pm}_{s+1}, P^{d\pm}_n, \bar{P}^{d\pm}_{s+1}, \bar{P}^{d\pm}_n

    This function corresponds to equations 81/82, 95/96, 103/104, A-8/A-9,
    A-24/A-25, and A-32/A-33 in [HuTS15]_, and loosely to the corresponding
    files ``Pdownmin.F90``, ``Pdownplus.F90``, ``Pupmin.F90``, and
    ``Pdownmin.F90``.

    This function is called from the function :mod:`kernel.greenfct`.

    """

    # Variables
    nlsr = abs(lsrc-lrec)+1  # nr of layers btw and incl. src and rec layer
    rsrcl = 0  # src-layer in reflection (Rp/Rm), first if down
    izrange = range(2, nlsr)
    isr = lsrc
    last = depth.size-1

    # Booleans if src in first or last layer; swapped if up=True
    first_layer = lsrc == 0
    last_layer = lsrc == depth.size-1

    # Depths; dp and dm are swapped if up=True
    if lsrc != depth.size-1:
        ds = depth[lsrc+1]-depth[lsrc]
        dp = depth[lsrc+1]-zsrc
    dm = zsrc-depth[lsrc]

    # Rm and Rp; swapped if up=True
    Rmp = Rm
    Rpm = Rp

    # Boolean if plus or minus has to be calculated
    plusset = [13, 23, 33, 14, 24, 34, 15, 25, 35]
    if TM:
        plus = ab in plusset
    else:
        plus = ab not in plusset

    # Sign-switches
    pm = 1     # + if plus=True, - if plus=False
    if not plus:
        pm = -1
    pup = -1   # + if up=True,   - if up=False
    mupm = 1   # + except if up=True and plus=False

    # Gamma of source layer
    iGam = Gam[:, :, lsrc, :]

    # Calculate down- and up-going fields
    for up in [False, True]:

        # No upgoing field if rec is in last layer or below src
        if up and (lrec == depth.size-1 or lrec > lsrc):
            Pu = np.zeros_like(iGam)
            continue
        # No downgoing field if rec is in first layer or above src
        if not up and (lrec == 0 or lrec < lsrc):
            Pd = np.zeros_like(iGam)
            continue

        # Swaps if up=True
        if up:
            if not last_layer:
                dp, dm = dm, dp
            else:
                dp = dm
            Rmp, Rpm = Rpm, Rmp
            first_layer, last_layer = last_layer, first_layer
            rsrcl = nlsr-1  # src-layer in refl. (Rp/Rm), last (nlsr-1) if up
            izrange = range(nlsr-2)
            isr = lrec
            last = 0
            pup = 1
            if not plus:
                mupm = -1

        # Calculate Pu+, Pu-, Pd+, Pd-
        if lsrc == lrec:  # rec in src layer; Eqs  81/82, A-8/A-9
            if last_layer:  # If src/rec are in top (up) or bottom (down) layer
                if use_ne_eval:
                    P = use_ne_eval("Rmp*exp(-iGam*dm)")
                else:
                    P = Rmp*np.exp(-iGam*dm)
            else:           # If src and rec are in any layer in between
                if use_ne_eval:
                    Pstr = "(exp(-iGam*dm) + pm*Rpm*exp(-iGam*(ds+dp)))"
                    Pstr += "* Rmp/(1-Rmp*Rpm*exp(-2*iGam*ds))"
                    P = use_ne_eval(Pstr)
                else:
                    P = np.exp(-iGam*dm) + pm*Rpm*np.exp(-iGam*(ds+dp))
                    P *= Rmp/(1 - Rmp*Rpm*np.exp(-2*iGam*ds))

        else:           # rec above (up) / below (down) src layer
            #           # Eqs  95/96,  A-24/A-25 for rec above src layer
            #           # Eqs 103/104, A-32/A-33 for rec below src layer

            # First compute P_{s-1} (up) / P_{s+1} (down)
            iRpm = Rpm[:, :, rsrcl, :]
            if first_layer:  # If src is in bottom (up) / top (down) layer
                if use_ne_eval:
                    P = use_ne_eval("(1 + iRpm)*mupm*exp(-iGam*dp)")
                else:
                    P = (1 + iRpm)*mupm*np.exp(-iGam*dp)
            else:
                iRmp = Rmp[:, :, rsrcl, :]
                if use_ne_eval:
                    Pstr = "(mupm*exp(-iGam*dp) + "
                    Pstr += "pm*mupm*iRmp*exp(-iGam*(ds+dm)))"
                    Pstr += "*(1 + iRpm)/(1 - iRmp*iRpm * exp(-2*iGam*ds))"
                    P = use_ne_eval(Pstr)
                else:
                    P = mupm*np.exp(-iGam*dp)
                    P += pm*mupm*iRmp*np.exp(-iGam * (ds+dm))
                    P *= (1 + iRpm)/(1 - iRmp*iRpm * np.exp(-2*iGam*ds))

            # If up or down and src is in last but one layer
            if up or (not up and lsrc+1 < depth.size-1):
                ddepth = depth[lsrc+1-1*pup]-depth[lsrc-1*pup]
                iRpm = Rpm[:, :, rsrcl-1*pup, :]
                miGam = Gam[:, :, lsrc-1*pup, :]
                if use_ne_eval:
                    P = use_ne_eval("P/(1 + iRpm*exp(-2*miGam * ddepth))")
                else:
                    P /= (1 + iRpm*np.exp(-2*miGam * ddepth))

            # Second compute P for all other layers
            if nlsr > 2:
                for iz in izrange:
                    ddepth = depth[isr+iz+pup+1]-depth[isr+iz+pup]
                    iRpm = Rpm[:, :, iz+pup, :]
                    piGam = Gam[:, :, isr+iz+pup, :]
                    if use_ne_eval:
                        P = use_ne_eval("P*(1 + iRpm)*exp(-piGam * ddepth)")
                    else:
                        P *= (1 + iRpm)*np.exp(-piGam * ddepth)

                    # If rec/src NOT in first/last layer (up/down)
                    if isr+iz != last:
                        ddepth = depth[isr+iz+1] - depth[isr+iz]
                        iRpm = Rpm[:, :, iz, :]
                        piGam2 = Gam[:, :, isr+iz, :]
                        if use_ne_eval:
                            Pstr = "P/(1 + iRpm*exp(-2*piGam2 * ddepth))"
                            P = use_ne_eval(Pstr)
                        else:
                            P /= 1 + iRpm*np.exp(-2*piGam2 * ddepth)

        # Store P in Pu/Pd
        if up:
            Pu = P
        else:
            Pd = P

    # Return fields (up- and downgoing)
    return Pu, Pd


# Angle Factor

def angle_factor(angle, ab, msrc, mrec):
    r"""Return the angle-dependent factor.

    The whole calculation in the wavenumber domain is only a function of the
    distance between the source and the receiver, it is independent of the
    angel. The angle-dependency is this factor, which can be applied to the
    corresponding parts in the wavenumber or in the frequency domain.

    The ``angle_factor`` corresponds to the sine and cosine-functions in Eqs
    105-107, 111-116, 119-121, 123-128.

    This function is called from one of the Hankel functions in
    :mod:`transform`.  Consult the modelling routines in :mod:`model` for a
    description of the input and output parameters.

    """

    # 33/66 are completely symmetric and hence independent of angle
    if ab in [33, ]:
        return np.ones(angle.size)

    # Evaluation angle
    eval_angle = angle.copy()

    # Add pi if receiver is magnetic (reciprocity), but not if source is
    # electric, because then source and receiver are swapped, ME => EM:
    # G^me_ab(s, r, e, z) = -G^em_ba(r, s, e, z).
    if mrec and not msrc:
        eval_angle += np.pi

    # Define fct (cos/sin) and angles to be tested
    if ab in [11, 22, 15, 24, 13, 31, 26, 35]:
        fct = np.cos
        test_ang_1 = np.pi/2
        test_ang_2 = 3*np.pi/2
    else:
        fct = np.sin
        test_ang_1 = np.pi
        test_ang_2 = 2*np.pi

    if ab in [11, 22, 15, 24, 12, 21, 14, 25]:
        eval_angle *= 2

    # Get factor
    factAng = fct(eval_angle)

    # Ensure cos([pi/2, 3pi/2]) and sin([pi, 2pi]) are zero (floating pt issue)
    factAng[np.isclose(np.abs(eval_angle), test_ang_1, 1e-10, 1e-14)] = 0
    factAng[np.isclose(np.abs(eval_angle), test_ang_2, 1e-10, 1e-14)] = 0

    return factAng


# Analytical solutions

def fullspace(off, angle, zsrc, zrec, etaH, etaV, zetaH, zetaV, ab, msrc,
              mrec):
    r"""Analytical full-space solutions in the frequency domain.

    .. math:: \hat{G}^{ee}_{\alpha\beta}, \hat{G}^{ee}_{3\alpha},
              \hat{G}^{ee}_{33}, \hat{G}^{em}_{\alpha\beta},
              \hat{G}^{em}_{\alpha 3}

    This function corresponds to equations 45--50 in [HuTS15]_, and loosely to
    the corresponding files ``Gin11.F90``, ``Gin12.F90``, ``Gin13.F90``,
    ``Gin22.F90``, ``Gin23.F90``, ``Gin31.F90``, ``Gin32.F90``, ``Gin33.F90``,
    ``Gin41.F90``, ``Gin42.F90``, ``Gin43.F90``, ``Gin51.F90``, ``Gin52.F90``,
    ``Gin53.F90``, ``Gin61.F90``, and ``Gin62.F90``.

    This function is called from one of the modelling routines in :mod:`model`.
    Consult these modelling routines for a description of the input and output
    parameters.

    """
    xco = np.cos(angle)*off
    yco = np.sin(angle)*off

    # Reciprocity switches for magnetic receivers
    if mrec:
        if msrc:  # If src is also magnetic, switch eta and zeta (MM => EE).
            # G^mm_ab(s, r, e, z) = -G^ee_ab(s, r, -z, -e)
            etaH, zetaH = -zetaH, -etaH
            etaV, zetaV = -zetaV, -etaV
        else:  # If src is electric, swap src and rec (ME => EM).
            # G^me_ab(s, r, e, z) = -G^em_ba(r, s, e, z)
            xco *= -1
            yco *= -1
            zsrc, zrec = zrec, zsrc

    # Calculate TE/TM-variables
    if ab not in [16, 26]:                      # Calc TM
        lGamTM = np.sqrt(zetaH*etaV)
        RTM = np.sqrt(off*off + ((zsrc-zrec)*(zsrc-zrec)*etaH/etaV)[:, None])
        uGamTM = np.exp(-lGamTM[:, None]*RTM)/(4*np.pi*RTM *
                                               np.sqrt(etaH/etaV)[:, None])

    if ab not in [13, 23, 31, 32, 33, 34, 35]:  # Calc TE
        lGamTE = np.sqrt(zetaV*etaH)
        RTE = np.sqrt(off*off+(zsrc-zrec)*(zsrc-zrec)*(zetaH/zetaV)[:, None])
        uGamTE = np.exp(-lGamTE[:, None]*RTE)/(4*np.pi*RTE *
                                               np.sqrt(zetaH/zetaV)[:, None])

    # Calculate responses
    if ab in [11, 12, 21, 22]:  # Eqs 45, 46

        # Define coo1, coo2, and delta
        if ab in [11, 22]:
            if ab in [11, ]:
                coo1 = xco
                coo2 = xco
            else:
                coo1 = yco
                coo2 = yco
            delta = 1
        else:
            coo1 = xco
            coo2 = yco
            delta = 0

        # Calculate response
        term1 = uGamTM*(3*coo1*coo2/(RTM*RTM) - delta)
        term1 *= 1/(etaV[:, None]*RTM*RTM) + (lGamTM/etaV)[:, None]/RTM
        term1 += uGamTM*zetaH[:, None]*coo1*coo2/(RTM*RTM)

        term2 = -delta*zetaH[:, None]*uGamTE

        term3 = -zetaH[:, None]*coo1*coo2/(off*off)*(uGamTM - uGamTE)

        term4 = -np.sqrt(zetaH)[:, None]*(2*coo1*coo2/(off*off) - delta)
        if np.any(zetaH.imag < 0):  # We need the sqrt where Im > 0.
            term4 *= -1     # This if-statement corrects for it.
        term4 *= np.exp(-lGamTM[:, None]*RTM) - np.exp(-lGamTE[:, None]*RTE)
        term4 /= 4*np.pi*np.sqrt(etaH)[:, None]*off*off

        gin = term1 + term2 + term3 + term4

    elif ab in [13, 23, 31, 32]:  # Eq 47

        # Define coo
        if ab in [13, 31]:
            coo = xco
        elif ab in [23, 32]:
            coo = yco

        # Calculate response
        term1 = (etaH/etaV)[:, None]*(zrec - zsrc)*coo/(RTM*RTM)
        term2 = 3/(RTM*RTM) + 3*lGamTM[:, None]/RTM + (lGamTM*lGamTM)[:, None]
        gin = term1*term2*uGamTM/etaV[:, None]

    elif ab in [33, ]:  # Eq 48

        # Calculate response
        term1 = (((etaH/etaV)[:, None]*(zsrc - zrec)/RTM) *
                 ((etaH/etaV)[:, None]*(zsrc - zrec)/RTM) *
                 (3/(RTM*RTM) + 3*lGamTM[:, None]/RTM +
                     (lGamTM*lGamTM)[:, None]))
        term2 = (-(etaH/etaV)[:, None]/RTM*(1/RTM + lGamTM[:, None]) -
                 (etaH*zetaH)[:, None])
        gin = (term1 + term2)*uGamTM/etaV[:, None]

    elif ab in [14, 24, 15, 25]:  # Eq 49

        # Define coo1, coo2, coo3, coo4, delta, and pm
        if ab in [14, 25]:
            coo1, coo2 = xco, yco
            coo3, coo4 = xco, yco
            delta = 0
            pm = -1
        elif ab in [24, 15]:
            coo1, coo2 = yco, yco
            coo3, coo4 = xco, xco
            delta = 1
            pm = 1

        # 15/25: Swap x/y
        if ab in[15, 25]:
            coo1, coo3 = coo3, coo1
            coo2, coo4 = coo4, coo2

        # 24/25: Swap src/rec
        if ab in[24, 25]:
            zrec, zsrc = zsrc, zrec

        # Calculate response
        def term(lGam, z_eH, z_eV, R, off, co1, co2):
            fac = (lGam*z_eH/z_eV)[:, None]/R*np.exp(-lGam[:, None]*R)
            term = 2/(off*off) + lGam[:, None]/R + 1/(R*R)
            return fac*(co1*co2*term - delta)
        termTM = term(lGamTM, etaH, etaV, RTM, off, coo1, coo2)
        termTE = term(lGamTE, zetaH, zetaV, RTE, off, coo3, coo4)
        mult = (zrec - zsrc)/(4*np.pi*np.sqrt(etaH*zetaH)[:, None]*off*off)
        gin = -mult*(pm*termTM + termTE)

    elif ab in [34, 35, 16, 26]:  # Eqs 50, 51

        # Define coo
        if ab in [34, 16]:
            coo = yco
        else:
            coo = -xco

        # Define R, lGam, uGam, e_zH, and e_zV
        if ab in [34, 35]:
            coo *= -1
            R = RTM
            lGam = lGamTM
            uGam = uGamTM
            e_zH = etaH
            e_zV = etaV
        else:
            R = RTE
            lGam = lGamTE
            uGam = uGamTE
            e_zH = zetaH
            e_zV = zetaV

        # Calculate response
        gin = coo*(e_zH/e_zV)[:, None]/R*(lGam[:, None] + 1/R)*uGam

    # If rec is magnetic switch sign (reciprocity MM/ME => EE/EM).
    if mrec:
        gin *= -1

    return gin


def halfspace(off, angle, zsrc, zrec, etaH, etaV, freqtime, ab, signal,
              solution='dhs'):
    r"""Return frequency- or time-space domain VTI half-space solution.

    Calculates the frequency- or time-space domain electromagnetic response for
    a half-space below air using the diffusive approximation, as given in
    [SlHM10]_, where the electric source is located at [x=0, y=0, z=zsrc>=0],
    and the electric receiver at [x=cos(angle)*off, y=sin(angle)*off,
    z=zrec>=0].

    It can also be used to calculate the fullspace solution or the separate
    fields: direct field, reflected field, and airwave; always using the
    diffusive approximation. See ``solution``-parameter.

    This function is called from one of the modelling routines in :mod:`model`.
    Consult these modelling routines for a description of the input and
    solution parameters.

    """
    xco = np.cos(angle)*off
    yco = np.sin(angle)*off
    res = np.real(1/etaH[0, 0])
    aniso = 1/np.sqrt(np.real(etaV[0, 0])*res)

    # Define sval/time and dtype depending on signal.
    if signal is None:
        sval = freqtime
        dtype = etaH.dtype
    else:
        time = freqtime
        if signal == -1:  # Calculate DC
            time = np.r_[time[:, 0], 1e4][:, None]
            freqtime = time
        dtype = float

    # Other defined parameters
    rh = np.sqrt(xco**2 + yco**2)  # Horizontal distance in space
    hp = abs(zrec + zsrc)          # Physical vertical distance
    hm = abs(zrec - zsrc)
    hsp = hp*aniso                 # Scaled vertical distance
    hsm = hm*aniso
    rp = np.sqrt(xco**2 + yco**2 + hp**2)    # Physical distance
    rm = np.sqrt(xco**2 + yco**2 + hm**2)
    rsp = np.sqrt(xco**2 + yco**2 + hsp**2)  # Scaled distance
    rsm = np.sqrt(xco**2 + yco**2 + hsm**2)
    #
    mu_0 = 4e-7*np.pi                   # Magn. perm. of free space  [H/m]
    tp = mu_0*rp**2/(res*4)             # Diffusion time
    tm = mu_0*rm**2/(res*4)
    tsp = mu_0*rsp**2/(res*aniso**2*4)  # Scaled diffusion time
    tsm = mu_0*rsm**2/(res*aniso**2*4)

    # delta-fct delta_\alpha\beta
    if ab in [11, 22, 33]:
        delta = 1
    else:
        delta = 0

    # Define alpha/beta; swap if necessary
    x = xco
    y = yco
    if ab == 11:
        y = x
    elif ab in [22, 23, 32]:
        x = y
    elif ab == 21:
        x, y = y, x

    # Define rev for 3\alpha->\alpha3 reciprocity
    if ab in [13, 23]:
        rev = -1
    elif ab in [31, 32]:
        rev = 1

    # Exponential diffusion functions for m=0,1,2

    if signal is None:  # Frequency-domain
        f0p = np.exp(-2*np.sqrt(sval*tp))
        f0m = np.exp(-2*np.sqrt(sval*tm))
        fs0p = np.exp(-2*np.sqrt(sval*tsp))
        fs0m = np.exp(-2*np.sqrt(sval*tsm))

        f1p = np.sqrt(sval)*f0p
        f1m = np.sqrt(sval)*f0m
        fs1p = np.sqrt(sval)*fs0p
        fs1m = np.sqrt(sval)*fs0m

        f2p = sval*f0p
        f2m = sval*f0m
        fs2p = sval*fs0p
        fs2m = sval*fs0m

    elif abs(signal) == 1:  # Time-domain step response
        # Replace F(m) with F(m-2)
        f0p = special.erfc(np.sqrt(tp/time))
        f0m = special.erfc(np.sqrt(tm/time))
        fs0p = special.erfc(np.sqrt(tsp/time))
        fs0m = special.erfc(np.sqrt(tsm/time))

        f1p = np.exp(-tp/time)/np.sqrt(np.pi*time)
        f1m = np.exp(-tm/time)/np.sqrt(np.pi*time)
        fs1p = np.exp(-tsp/time)/np.sqrt(np.pi*time)
        fs1m = np.exp(-tsm/time)/np.sqrt(np.pi*time)

        f2p = f1p*np.sqrt(tp)/time
        f2m = f1m*np.sqrt(tm)/time
        fs2p = fs1p*np.sqrt(tsp)/time
        fs2m = fs1m*np.sqrt(tsm)/time

    else:  # Time-domain impulse response
        f0p = np.sqrt(tp/(np.pi*time**3))*np.exp(-tp/time)
        f0m = np.sqrt(tm/(np.pi*time**3))*np.exp(-tm/time)
        fs0p = np.sqrt(tsp/(np.pi*time**3))*np.exp(-tsp/time)
        fs0m = np.sqrt(tsm/(np.pi*time**3))*np.exp(-tsm/time)

        f1p = (tp/time - 0.5)/np.sqrt(tp)*f0p
        f1m = (tm/time - 0.5)/np.sqrt(tm)*f0m
        fs1p = (tsp/time - 0.5)/np.sqrt(tsp)*fs0p
        fs1m = (tsm/time - 0.5)/np.sqrt(tsm)*fs0m

        f2p = (tp/time - 1.5)/time*f0p
        f2m = (tm/time - 1.5)/time*f0m
        fs2p = (tsp/time - 1.5)/time*fs0p
        fs2m = (tsm/time - 1.5)/time*fs0m

    # Pre-allocate arrays
    gs0m = np.zeros(np.shape(x), dtype=dtype)
    gs0p = np.zeros(np.shape(x), dtype=dtype)
    gs1m = np.zeros(np.shape(x), dtype=dtype)
    gs1p = np.zeros(np.shape(x), dtype=dtype)
    gs2m = np.zeros(np.shape(x), dtype=dtype)
    gs2p = np.zeros(np.shape(x), dtype=dtype)
    g0p = np.zeros(np.shape(x), dtype=dtype)
    g1m = np.zeros(np.shape(x), dtype=dtype)
    g1p = np.zeros(np.shape(x), dtype=dtype)
    g2m = np.zeros(np.shape(x), dtype=dtype)
    g2p = np.zeros(np.shape(x), dtype=dtype)
    air = np.zeros(np.shape(f0p), dtype=dtype)

    if ab in [11, 12, 21, 22]:  # 1. {alpha, beta}
        # Get indices for singularities
        izr = rh == 0         # index where rh = 0
        iir = np.invert(izr)  # invert of izr
        izh = hm == 0         # index where hm = 0
        iih = np.invert(izh)  # invert of izh

        # fab
        fab = rh**2*delta-x*y

        # TM-mode coefficients
        gs0p = res*aniso*(3*x*y - rsp**2*delta)/(4*np.pi*rsp**5)
        gs0m = res*aniso*(3*x*y - rsm**2*delta)/(4*np.pi*rsm**5)
        gs1p[iir] = (((3*x[iir]*y[iir] - rsp[iir]**2*delta)/rsp[iir]**4 -
                     (x[iir]*y[iir] - fab[iir])/rh[iir]**4) *
                     np.sqrt(mu_0*res)/(4*np.pi))
        gs1m[iir] = (((3*x[iir]*y[iir] - rsm[iir]**2*delta)/rsm[iir]**4 -
                     (x[iir]*y[iir] - fab[iir])/rh[iir]**4) *
                     np.sqrt(mu_0*res)/(4*np.pi))
        gs2p[iir] = ((mu_0*x[iir]*y[iir])/(4*np.pi*aniso*rsp[iir]) *
                     (1/rsp[iir]**2 - 1/rh[iir]**2))
        gs2m[iir] = ((mu_0*x[iir]*y[iir])/(4*np.pi*aniso*rsm[iir]) *
                     (1/rsm[iir]**2 - 1/rh[iir]**2))

        # TM-mode for numerical singularities rh=0 (hm!=0)
        gs1p[izr*iih] = -np.sqrt(mu_0*res)*delta/(4*np.pi*hsp**2)
        gs1m[izr*iih] = -np.sqrt(mu_0*res)*delta/(4*np.pi*hsm**2)
        gs2p[izr*iih] = -mu_0*delta/(8*np.pi*aniso*hsp)
        gs2m[izr*iih] = -mu_0*delta/(8*np.pi*aniso*hsm)

        # TE-mode coefficients
        g0p = res*(3*fab - rp**2*delta)/(2*np.pi*rp**5)
        g1m[iir] = (np.sqrt(mu_0*res)*(x[iir]*y[iir] - fab[iir]) /
                    (4*np.pi*rh[iir]**4))
        g1p[iir] = (g1m[iir] + np.sqrt(mu_0*res)*(3*fab[iir] -
                    rp[iir]**2*delta)/(2*np.pi*rp[iir]**4))
        g2p[iir] = mu_0*fab[iir]/(4*np.pi*rp[iir])*(2/rp[iir]**2 -
                                                    1/rh[iir]**2)
        g2m[iir] = -mu_0*fab[iir]/(4*np.pi*rh[iir]**2*rm[iir])

        # TE-mode for numerical singularities rh=0 (hm!=0)
        g1m[izr*iih] = np.zeros(np.shape(g1m[izr*iih]), dtype=dtype)
        g1p[izr*iih] = -np.sqrt(mu_0*res)*delta/(2*np.pi*hp**2)
        g2m[izr*iih] = mu_0*delta/(8*np.pi*hm)
        g2p[izr*iih] = mu_0*delta/(8*np.pi*hp)

        # Bessel functions for airwave
        def BI(gamH, hp, nr, xim):
            r"""Return BI_nr."""
            return np.exp(-np.real(gamH)*hp)*special.ive(nr, xim)

        def BK(xip, nr):
            r"""Return BK_nr."""
            if np.isrealobj(xip):
                # To keep it real in Laplace-domain [exp(-1j*0) = 1-0j].
                return special.kve(nr, xip)
            else:
                return np.exp(-1j*np.imag(xip))*special.kve(nr, xip)

        # Airwave calculation
        def airwave(sval, hp, rp, res, fab, delta):
            r"""Return airwave."""
            # Parameters
            zeta = sval*mu_0
            gamH = np.sqrt(zeta/res)
            xip = gamH*(rp + hp)/2
            xim = gamH*(rp - hp)/2

            # Bessel functions
            BI0 = BI(gamH, hp, 0, xim)
            BI1 = BI(gamH, hp, 1, xim)
            BI2 = BI(gamH, hp, 2, xim)
            BK0 = BK(xip, 0)
            BK1 = BK(xip, 1)

            # Calculation
            P1 = (sval*mu_0)**(3/2)*fab*hp/(4*np.sqrt(res))
            P2 = 4*BI1*BK0 - (3*BI0 - 4*np.sqrt(res)*BI1/(np.sqrt(sval*mu_0) *
                              (rp + hp)) + BI2)*BK1
            P3 = 3*fab/rp**2 - delta
            P4 = (sval*mu_0*hp*rp*(BI0*BK0 - BI1*BK1) +
                  np.sqrt(res*sval*mu_0)*BI0*BK1 *
                  (rp + hp) + np.sqrt(res*sval*mu_0)*BI1*BK0*(rp - hp))

            return (P1*P2 - P3*P4)/(4*np.pi*rp**3)

        # Airwave depending on signal
        if signal is None:  # Frequency-domain
            air = airwave(sval, hp, rp, res, fab, delta)

        elif abs(signal) == 1:  # Time-domain step response
            # Solution for step-response air-wave is not analytical, but uses
            # the Gaver-Stehfest method.
            K = 16

            # Coefficients Dk
            fn = np.vectorize(np.math.factorial)

            def coeff_dk(k, K):
                r"""Return coefficients Dk for k, K."""
                n = np.arange(int((k+1)/2), np.min([k, K/2])+.5, 1, dtype=int)
                Dk = n**(K/2)*fn(2*n)
                Dk /= fn(n)*fn(n-1)*fn(k-n)*fn(2*n-k)*fn(K/2-n)
                return Dk.sum()*(-1)**(k+K/2)

            for k in range(1, K+1):
                sval = k*np.log(2)/time
                cair = airwave(sval, hp, rp, res, fab, delta)
                air += coeff_dk(k, K)*cair.real/k

        else:  # Time-domain impulse response
            thp = mu_0*hp**2/(4*res)
            trh = mu_0*rh**2/(8*res)
            P1 = (mu_0**2*hp*np.exp(-thp/time))/(res*32*np.pi*time**3)
            P2 = 2*(delta - (x*y)/rh**2)*special.ive(1, trh/time)
            P3 = mu_0/(2*res*time)*(rh**2*delta - x*y)-delta
            P4 = special.ive(0, trh/time) - special.ive(1, trh/time)

            air = P1*(P2 - P3*P4)

    elif ab in [13, 23, 31, 32]:  # 2. {3, alpha}, {alpha, 3}
        # TM-mode
        gs0m = 3*x*res*aniso**3*(zrec - zsrc)/(4*np.pi*rsm**5)
        gs0p = rev*3*x*res*aniso**3*hp/(4*np.pi*rsp**5)
        gs1m = (np.sqrt(mu_0*res)*3*aniso**2*x*(zrec - zsrc) /
                (4*np.pi*rsm**4))
        gs1p = rev*np.sqrt(mu_0*res)*3*aniso**2*x*hp/(4*np.pi*rsp**4)
        gs2m = mu_0*x*aniso*(zrec - zsrc)/(4*np.pi*rsm**3)
        gs2p = rev*mu_0*x*aniso*hp/(4*np.pi*rsp**3)

    elif ab == 33:  # 3. {3, 3}
        # TM-mode
        gs0m = res*aniso**3*(3*hsm**2 - rsm**2)/(4*np.pi*rsm**5)
        gs0p = -res*aniso**3*(3*hsp**2 - rsp**2)/(4*np.pi*rsp**5)
        gs1m = np.sqrt(mu_0*res)*aniso**2*(3*hsm**2 - rsm**2)/(4*np.pi*rsm**4)
        gs1p = -np.sqrt(mu_0*res)*aniso**2*(3*hsp**2 - rsp**2)/(4*np.pi*rsp**4)
        gs2m = mu_0*aniso*(hsm**2 - rsm**2)/(4*np.pi*rsm**3)
        gs2p = -mu_0*aniso*(hsp**2 - rsp**2)/(4*np.pi*rsp**3)

    # Direct field
    direct_TM = gs0m*fs0m + gs1m*fs1m + gs2m*fs2m
    direct_TE = g1m*f1m + g2m*f2m
    direct = direct_TM + direct_TE

    # Reflection
    reflect_TM = gs0p*fs0p + gs1p*fs1p + gs2p*fs2p
    reflect_TE = g0p*f0p + g1p*f1p + g2p*f2p
    reflect = reflect_TM + reflect_TE

    # If switch-off, subtract switch-on from DC value
    if signal == -1:
        direct_TM = direct_TM[-1]-direct_TM[:-1]
        direct_TE = direct_TE[-1]-direct_TE[:-1]
        direct = direct[-1]-direct[:-1]

        reflect_TM = reflect_TM[-1]-reflect_TM[:-1]
        reflect_TE = reflect_TE[-1]-reflect_TE[:-1]
        reflect = reflect[-1]-reflect[:-1]

        air = air[-1]-air[:-1]

    # Return, depending on 'solution'
    if solution == 'dfs':
        return direct
    elif solution == 'dsplit':
        return direct, reflect, air
    elif solution == 'dtetm':
        return direct_TE, direct_TM, reflect_TE, reflect_TM, air
    else:
        return direct + reflect + air
