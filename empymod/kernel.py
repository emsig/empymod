"""
Kernel of empymod, calculates the wavenumber-domain electromagnetic
response. Plus analytical full- and half-space solutions.

The functions :func:`wavenumber`, :func:`angle_factor`, :func:`fullspace`,
:func:`greenfct`, :func:`reflections`, and :func:`fields` are based on source
files (specified in each function) from the source code distributed with
[HuTS15]_, which can be found at `software.seg.org/2015/0001
<https://software.seg.org/2015/0001>`_.  These functions are (c) 2015 by
Hunziker et al. and the Society of Exploration Geophysicists,
https://software.seg.org/disclaimer.txt.  Please read the NOTICE-file in the
root directory for more information regarding the involved licenses.

"""
# Copyright 2016 The emsig community.
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
import scipy as sp
import numba as nb

__all__ = ['wavenumber', 'angle_factor', 'fullspace', 'greenfct',
           'reflections', 'fields', 'halfspace']

# Numba-settings
_numba_setting = {'nogil': True, 'cache': True}
_numba_with_fm = {'fastmath': True, **_numba_setting}
nb.config.DISABLE_JIT = True  # Disable JIT for testing


def __dir__():
    return __all__


# Wavenumber-frequency domain kernel

@nb.njit(**_numba_setting)
def wavenumber(zsrc, zrec, lsrc, lrec, depth, etaH, etaV, zetaH, zetaV, lambd,
               ab, xdirect, msrc, mrec, ana_deriv: bool = False):
    r"""Calculate wavenumber domain solution.

    Return the wavenumber domain solutions `PJ0`, `PJ1`, and `PJ0b`, which have
    to be transformed with a Hankel transform to the frequency domain.
    `PJ0`/`PJ0b` and `PJ1` have to be transformed with Bessel functions of
    order 0 (:math:`J_0`) and 1 (:math:`J_1`), respectively.

    This function corresponds loosely to equations 105--107, 111--116,
    119--121, and 123--128 in [HuTS15]_, and equally loosely to the file
    `kxwmod.c`.

    [HuTS15]_ uses Bessel functions of orders 0, 1, and 2 (:math:`J_0, J_1,
    J_2`). The implementations of the *Fast Hankel Transform* and the
    *Quadrature-with-Extrapolation* in :mod:`empymod.transform` are set-up with
    Bessel functions of order 0 and 1 only. This is achieved by applying the
    recurrence formula

    .. math::
        :label: wavenumber

        J_2(kr) = \frac{2}{kr} J_1(kr) - J_0(kr) \ .


    .. note::

        `PJ0` and `PJ0b` could theoretically be added here into one, and then
        be transformed in one go.  However, `PJ0b` has to be multiplied by
        :func:`ang_fact` later. This has to be done after the Hankel transform
        for methods which make use of spline interpolation, in order to work
        for offsets that are not in line with each other.

    This function is called from one of the Hankel functions in
    :mod:`empymod.transform`.  Consult the modelling routines in
    :mod:`empymod.model` for a description of the input and output parameters.

    If you are solely interested in the wavenumber-domain solution you can call
    this function directly. However, you have to make sure all input arguments
    are correct, as no checks are carried out here.

    """
    nfreq, nlayer = etaH.shape
    noff, nlambda = lambd.shape

    # ** CALCULATE GREEN'S FUNCTIONS
    # Shape of PTM, PTE: (nfreq, noffs, nfilt)
    if not ana_deriv:
        PTM, PTE = greenfct(zsrc, zrec, lsrc, lrec, depth, etaH, etaV, zetaH,
                            zetaV, lambd, ab, xdirect, msrc, mrec, )
    else:
        PTM, PTE, dPTM, dPTE = greenfct(zsrc, zrec, lsrc, lrec, depth, etaH,
                                        etaV, zetaH, zetaV, lambd, ab, xdirect,
                                        msrc, mrec, ana_deriv=True)

    # ** AB-SPECIFIC COLLECTION OF PJ0, PJ1, AND PJ0b

    # Pre-allocate output
    if ab in [11, 22, 24, 15, 33]:
        PJ0 = np.zeros_like(PTM)
        if ana_deriv:
            dPJ0 = np.zeros((nfreq, noff, nlambda, nlayer), dtype=PTM.dtype)
    else:
        PJ0 = None
    if ab in [11, 12, 21, 22, 14, 24, 15, 25]:
        PJ0b = np.zeros_like(PTM)
        if ana_deriv:
            dPJ0b = np.zeros((nfreq, noff, nlambda, nlayer), dtype=PTM.dtype)
    else:
        PJ0b = None
    if ab not in [33, ]:
        PJ1 = np.zeros_like(PTM)
        if ana_deriv:
            dPJ1 = np.zeros((nfreq, noff, nlambda, nlayer), dtype=PTM.dtype)
    else:
        PJ1 = None
    Ptot = np.zeros_like(PTM)
    if ana_deriv:
        dPtot = np.zeros((nfreq, noff, nlambda, nlayer), dtype=PTM.dtype)

    # Calculate Ptot which is used in all cases
    fourpi = 4*np.pi
    for i in range(nfreq):
        for ii in range(noff):
            for iv in range(nlambda):
                Ptot[i, ii, iv] = (PTM[i, ii, iv] + PTE[i, ii, iv])/fourpi
                if ana_deriv:
                    for v in range(nlayer):
                        dPtot[i, ii, iv, v] = (dPTM[i, ii, iv, v] + dPTE[i, ii, iv, v])/fourpi

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

        for i in range(nfreq):
            for ii in range(noff):
                for iv in range(nlambda):
                    PJ0b[i, ii, iv] = sign/2*Ptot[i, ii, iv]*lambd[ii, iv]
                    PJ1[i, ii, iv] = -sign*Ptot[i, ii, iv]
                    if ana_deriv:
                        for v in range(nlayer):
                            dPJ0b[i, ii, iv, v] = sign/2*dPtot[i, ii, iv, v]*lambd[ii, iv]
                            dPJ1[i, ii, iv, v] = -sign*dPtot[i, ii, iv, v]

        if ab in [11, 22, 24, 15]:
            if ab in [22, 24]:
                sign *= -1

            eightpi = sign*8*np.pi
            for i in range(nfreq):
                for ii in range(noff):
                    for iv in range(nlambda):
                        PJ0[i, ii, iv] = PTM[i, ii, iv] - PTE[i, ii, iv]
                        PJ0[i, ii, iv] *= lambd[ii, iv]/eightpi
                        if ana_deriv:
                            for v in range(nlayer):
                                dPJ0[i, ii, iv, v] = dPTM[i, ii, iv, v] - dPTE[i, ii, iv, v]
                                dPJ0[i, ii, iv, v] *= lambd[ii, iv]/eightpi

    elif ab in [13, 23, 31, 32, 34, 35, 16, 26]:  # Eqs 107, 113, 114, 115,
        if ab in [34, 26]:                        # .   121, 125, 126, 127
            sign *= -1
        for i in range(nfreq):
            for ii in range(noff):
                for iv in range(nlambda):
                    dlambd = lambd[ii, iv]*lambd[ii, iv]
                    PJ1[i, ii, iv] = sign*Ptot[i, ii, iv]*dlambd
                    if ana_deriv:
                        for v in range(nlayer):
                            dPJ1[i, ii, iv, v] = sign*dPtot[i, ii, iv, v]*dlambd

    elif ab in [33, ]:                            # Eq 116
        for i in range(nfreq):
            for ii in range(noff):
                for iv in range(nlambda):
                    tlambd = lambd[ii, iv]*lambd[ii, iv]*lambd[ii, iv]
                    PJ0[i, ii, iv] = sign*Ptot[i, ii, iv]*tlambd
                    if ana_deriv:
                        for v in range(nlayer):
                            dPJ0[i, ii, iv, v] = sign*dPtot[i, ii, iv, v]*tlambd

    # Return PJ0, PJ1, PJ0b
    if not ana_deriv:
        return PJ0, PJ1, PJ0b
    else:
        return PJ0, PJ1, PJ0b, dPJ0, dPJ1, dPJ0b


@nb.njit(**_numba_setting)
def greenfct(zsrc, zrec, lsrc, lrec, depth, etaH, etaV, zetaH, zetaV, lambd,
             ab, xdirect, msrc, mrec, ana_deriv: bool = False, debug: bool = False):
    r"""Calculate Green's function for TM and TE.

    .. math::
        :label: greenfct

        \tilde{g}^{tm}_{hh}, \tilde{g}^{tm}_{hz},
        \tilde{g}^{tm}_{zh}, \tilde{g}^{tm}_{zz},
        \tilde{g}^{te}_{hh}, \tilde{g}^{te}_{zz}

    This function corresponds to equations 108--110, 117/118, 122; 89--94,
    A18--A23, B13--B15; 97--102 A26--A31, and B16--B18 in [HuTS15]_, and
    loosely to the corresponding files `Gamma.F90`, `Wprop.F90`, `Ptotalx.F90`,
    `Ptotalxm.F90`, `Ptotaly.F90`, `Ptotalym.F90`, `Ptotalz.F90`, and
    `Ptotalzm.F90`.

    The Green's functions are multiplied according to Eqs 105-107, 111-116,
    119-121, 123-128; with the factors inside the integrals.

    This function is called from the function :func:`wavenumber`.

    """
    nfreq, nlayer = etaH.shape
    noff, nlambda = lambd.shape

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
        Gam = np.zeros((nfreq, noff, nlayer, nlambda), etaH.dtype)
        for i in range(nfreq):
            for ii in range(noff):
                for iii in range(nlayer):
                    h_div_v = e_zH[i, iii]/e_zV[i, iii]
                    h_times_h = z_eH[i, iii]*e_zH[i, iii]
                    for iv in range(nlambda):
                        l2 = lambd[ii, iv]*lambd[ii, iv]
                        Gam[i, ii, iii, iv] = np.sqrt(h_div_v*l2 + h_times_h)

        # Gamma in receiver layer
        lrecGam = Gam[:, :, lrec, :]

        # Derivative of Gamma
        if ana_deriv:
            dGam = np.zeros_like(Gam, dtype=Gam.dtype)
            # TODO: Extend to VTI case, currently isotropic only
            if not np.array_equal(etaH, etaV):
                raise NotImplementedError("Analytical derivatives are only implemented for isotropic models.")
            if not np.array_equal(zetaH, zetaV):
                raise NotImplementedError("Analytical derivatives are only implemented for isotropic models.")
            for i in range(nfreq):
                for ii in range(noff):
                    for v in range(nlayer):
                        for iv in range(nlambda):
                            if TM:
                                dGam[i, ii, v, iv] = z_eH[i, v]/(2*Gam[i, ii, v, iv])
                            else:
                                dGam[i, ii, v, iv] = e_zH[i, v]/(2*Gam[i, ii, v, iv])
            # derivative of gamma in receiver layer
            lrecdGam = dGam[:, :, lrec, :]

        # Green's functions
        green = np.zeros_like(lrecGam)

        # Reflection (coming from below (Rp) and above (Rm) rec)
        if depth.size > 1:  # Only if more than 1 layer
            if ana_deriv:
                Rp, Rm, dRp, dRm = reflections(depth, e_zH, Gam, lrec, lsrc, ana_deriv=ana_deriv, dGam=dGam)
                # Field at rec level (coming from below (Pu) and above (Pd) rec)
                Pu, Pd, dPu, dPd = fields(depth, Rp, Rm, Gam, lrec, lsrc, zsrc, ab, TM, ana_deriv=ana_deriv, dRp=dRp,
                                          dRm=dRm, dGam=dGam)
                dgreen = np.zeros_like(Gam)
                dgreen = np.swapaxes(dgreen, 2, 3)  # for lrecGam, but number of layers for derivates as last axist
            else:
                Rp, Rm = reflections(depth, e_zH, Gam, lrec,
                                     lsrc)  # TODO: @Dieter, why switch here? reflections yiels Rm, Rp
                # Field at rec level (coming from below (Pu) and above (Pd) rec)
                Pu, Pd = fields(depth, Rp, Rm, Gam, lrec, lsrc, zsrc, ab, TM)

            # Field propagators
            # (Up- (Wu) and downgoing (Wd), in rec layer); Eq 74
            Wu = np.zeros_like(lrecGam)
            Wd = np.zeros_like(lrecGam)
            if ana_deriv:  # Same number of dimensions of Wu, as Wu_n only depends on sigma_n and not sigma_n-1
                dWu = np.zeros_like(
                    lrecGam)  # .swapaxes(2, 3)  # for lrecGam, but number of layers for derivates as last axist
                dWd = np.zeros_like(
                    lrecGam)  # .swapaxes(2, 3)  # for lrecGam, but number of layers for derivates as last axist

            if lrec != depth.size-1:  # No upgoing field prop. if rec in last
                ddepth = depth[lrec + 1] - zrec
                for i in range(nfreq):
                    for ii in range(noff):
                        for iv in range(nlambda):
                            Wu[i, ii, iv] = np.exp(-lrecGam[i, ii, iv]*ddepth)
                            if ana_deriv:
                                dWu[i, ii, iv] = -ddepth*Wu[i, ii, iv]*lrecdGam[i, ii, iv]

            if lrec != 0:     # No downgoing field propagator if rec in first
                ddepth = zrec - depth[lrec]
                for i in range(nfreq):
                    for ii in range(noff):
                        for iv in range(nlambda):
                            Wd[i, ii, iv] = np.exp(-lrecGam[i, ii, iv]*ddepth)
                            if ana_deriv:
                                dWd[i, ii, iv] = -ddepth*Wd[i, ii, iv]*lrecdGam[i, ii, iv]

        if lsrc == lrec:  # Rec in src layer; Eqs 108, 109, 110, 117, 118, 122

            # Green's function depending on <ab>
            # (If only one layer, no reflections/fields)
            if depth.size > 1 and ab in [13, 23, 31, 32, 14, 24, 15, 25]:
                for i in range(nfreq):
                    for ii in range(noff):
                        for iv in range(nlambda):
                            green[i, ii, iv] = Pu[i, ii, iv]*Wu[i, ii, iv]
                            green[i, ii, iv] -= Pd[i, ii, iv]*Wd[i, ii, iv]
                            if ana_deriv:
                                for v in range(nlayer):
                                    dgreen[i, ii, iv, v] = dPu[i, ii, iv, v]*Wu[i, ii, iv] - dPd[i, ii, iv, v]*Wd[
                                        i, ii, iv]
                                    if v == lrec:
                                        dgreen[i, ii, iv, v] += Pu[i, ii, iv]*dWu[i, ii, iv] - Pd[i, ii, iv]*dWd[
                                            i, ii, iv]



            elif depth.size > 1:
                for i in range(nfreq):
                    for ii in range(noff):
                        for iv in range(nlambda):
                            green[i, ii, iv] = Pu[i, ii, iv]*Wu[i, ii, iv]
                            green[i, ii, iv] += Pd[i, ii, iv]*Wd[i, ii, iv]
                            if ana_deriv:
                                for v in range(nlayer):
                                    dgreen[i, ii, iv, v] = dPu[i, ii, iv, v]*Wu[i, ii, iv] + dPd[i, ii, iv, v]*Wd[
                                        i, ii, iv]
                                    if v == lrec:
                                        dgreen[i, ii, iv, v] += Pu[i, ii, iv]*dWu[i, ii, iv] + Pd[i, ii, iv]*dWd[
                                            i, ii, iv]

            # Direct field, if it is computed in the wavenumber domain
            if not xdirect:
                ddepth = abs(zsrc - zrec)
                dsign = np.sign(zrec - zsrc)
                minus_ab = [11, 12, 13, 14, 15, 21, 22, 23, 24, 25]

                for i in range(nfreq):
                    for ii in range(noff):
                        for iv in range(nlambda):

                            # Direct field
                            directf = np.exp(-lrecGam[i, ii, iv]*ddepth)
                            if ana_deriv:
                                ddirectf = -ddepth*directf*lrecdGam[i, ii, iv]

                            # Swap TM for certain <ab>
                            if TM and ab in minus_ab:
                                directf *= -1
                                if ana_deriv:
                                    ddirectf *= -1

                            # Multiply by zrec-zsrc-sign for certain <ab>
                            if ab in [13, 14, 15, 23, 24, 25, 31, 32]:
                                directf *= dsign
                                if ana_deriv:
                                    ddirectf *= dsign

                            # Add direct field to Green's function
                            green[i, ii, iv] += directf
                            if ana_deriv:
                                dgreen[i, ii, iv, lrec] += ddirectf

        else:

            # Calculate exponential factor
            if lrec == depth.size-1:
                ddepth = 0
            else:
                ddepth = depth[lrec+1] - depth[lrec]

            fexp = np.zeros_like(lrecGam)
            for i in range(nfreq):
                for ii in range(noff):
                    for iv in range(nlambda):
                        fexp[i, ii, iv] = np.exp(-lrecGam[i, ii, iv]*ddepth)

            # Sign-switch for Green calculation
            if TM and ab in [11, 12, 13, 21, 22, 23, 14, 24, 15, 25]:
                pmw = -1
            else:
                pmw = 1

            if lrec < lsrc:  # Rec above src layer: Pd not used
                #              Eqs 89-94, A18-A23, B13-B15
                for i in range(nfreq):
                    for ii in range(noff):
                        for iv in range(nlambda):
                            green[i, ii, iv] = Pu[i, ii, iv]*(
                                    Wu[i, ii, iv] + pmw*Rm[i, ii, 0, iv] *
                                    fexp[i, ii, iv]*Wd[i, ii, iv])

            elif lrec > lsrc:  # rec below src layer: Pu not used
                #                Eqs 97-102 A26-A30, B16-B18
                for i in range(nfreq):
                    for ii in range(noff):
                        for iv in range(nlambda):
                            green[i, ii, iv] = Pd[i, ii, iv]*(
                                    pmw*Wd[i, ii, iv] +
                                    Rp[i, ii, abs(lsrc-lrec), iv] *
                                    fexp[i, ii, iv]*Wu[i, ii, iv])

        # Store in corresponding variable
        if TM:
            gamTM, GTM = Gam, green
            gTM = green.copy()
            if ana_deriv:
                dgamTM, dgTM = dGam, dgreen
                dGTM = np.zeros_like(dgTM)
        else:
            gamTE, GTE = Gam, green
            gTE = green.copy()
            if ana_deriv:
                dgamTE, dgTE = dGam, dgreen
                dGTE = np.zeros_like(dgTE)

    # ** AB-SPECIFIC FACTORS AND CALCULATION OF PTOT'S
    # These are the factors inside the integrals
    # Eqs 105-107, 111-116, 119-121, 123-128

    if ab in [11, 12, 21, 22]:
        for i in range(nfreq):
            for ii in range(noff):
                for iv in range(nlambda):
                    GTM[i, ii, iv] *= gamTM[i, ii, lrec, iv]/etaH[i, lrec]
                    GTE[i, ii, iv] *= zetaH[i, lsrc]/gamTE[i, ii, lsrc, iv]  # TODO: Why lrev vs lsrc?
                    if ana_deriv:
                        if lrec != lsrc:
                            raise NotImplementedError("Possible error lsrc vs lrec for TM vs TE")
                        for v in range(nlayer):
                            # TODO: Problem here: derivative of kernel in  eq. 105-107 -> apply chain rule, sum f dGTM twice so this approach cannot be used.
                            dGTM[i, ii, iv, v] = dgTM[i, ii, iv, v]*gamTM[i, ii, lrec, iv]/etaH[i, lrec]
                            dGTE[i, ii, iv, v] = dgTE[i, ii, iv, v]*zetaH[i, lsrc]/gamTE[i, ii, lsrc, iv]
                            # Extra term for s == n; dirac_sn
                            if v == lrec:
                                dGTM[i, ii, iv, v] += -0.5*(gamTM[i, ii, lrec, iv]/etaH[i, lrec])*gTM[i, ii, iv]
                            if v == lsrc:
                                dGTE[i, ii, iv, v] += -0.5*(zetaH[i, lsrc]/(gamTE[i, ii, lsrc, iv]*etaH[i, lsrc]))*gTE[
                                    i, ii, iv]

    elif ab in [14, 15, 24, 25]:
        for i in range(nfreq):
            fact = etaH[i, lsrc]/etaH[i, lrec]
            for ii in range(noff):
                for iv in range(nlambda):
                    GTM[i, ii, iv] *= fact*gamTM[i, ii, lrec, iv]
                    GTM[i, ii, iv] /= gamTM[i, ii, lsrc, iv]

    elif ab in [13, 23]:
        GTE = np.zeros_like(GTM)
        for i in range(nfreq):
            fact = etaH[i, lsrc]/etaH[i, lrec]/etaV[i, lsrc]
            for ii in range(noff):
                for iv in range(nlambda):
                    GTM[i, ii, iv] *= -fact*gamTM[i, ii, lrec, iv]
                    GTM[i, ii, iv] /= gamTM[i, ii, lsrc, iv]

    elif ab in [31, 32]:
        GTE = np.zeros_like(GTM)
        for i in range(nfreq):
            for ii in range(noff):
                for iv in range(nlambda):
                    GTM[i, ii, iv] /= etaV[i, lrec]

    elif ab in [34, 35]:
        GTE = np.zeros_like(GTM)
        for i in range(nfreq):
            fact = etaH[i, lsrc]/etaV[i, lrec]
            for ii in range(noff):
                for iv in range(nlambda):
                    GTM[i, ii, iv] *= fact/gamTM[i, ii, lsrc, iv]

    elif ab in [16, 26]:
        GTM = np.zeros_like(GTE)
        for i in range(nfreq):
            fact = zetaH[i, lsrc]/zetaV[i, lsrc]
            for ii in range(noff):
                for iv in range(nlambda):
                    GTE[i, ii, iv] *= fact/gamTE[i, ii, lsrc, iv]

    elif ab in [33, ]:
        GTE = np.zeros_like(GTM)
        for i in range(nfreq):
            fact = etaH[i, lsrc]/etaV[i, lsrc]/etaV[i, lrec]
            for ii in range(noff):
                for iv in range(nlambda):
                    GTM[i, ii, iv] *= fact/gamTM[i, ii, lsrc, iv]

    # Return Green's functions
    if debug:
        if ana_deriv:
            return gTM, gTE, dgTM, dgTE, gamTM, gamTE, dgamTM, dgamTE, Pu, dPu, Pd, dPd
        else:
            return gTM, gTE, gamTM, gamTE

    else:
        if ana_deriv:
            return GTM, GTE, dGTM, dGTE
        else:
            return GTM, GTE


@nb.njit(**_numba_with_fm)
def reflections(depth, e_zH, Gam, lrec, lsrc, ana_deriv: bool = False, dGam=None, debug=False):
    # TODO:
    #    - Currently only isotropic
    #    - Think about providing dGam instead to avoid code duplication.
    #    - Memory Efficiency: We could store the dRef in one big 5d tensor.
    #    Upper right part is the plus part, lewer left part is the minus part.
    #    Then two vectors needed for the diagonals. This would approx. halve the memory usage of dRef.
    r"""Calculate Rp, Rm.

    .. math::
        :label: reflections

        R^\pm_n, \bar{R}^\pm_n

    This function corresponds to equations 64/65 and A-11/A-12 in
    [HuTS15]_, and loosely to the corresponding files `Rmin.F90` and
    `Rplus.F90`.

    This function is called from the function :func:`greenfct`.

    """

    # Get numbers and max/min layer.
    nfreq, noff, nlayer, nlambda = Gam[:, :, :, :].shape
    maxl = max([lrec, lsrc])
    minl = min([lrec, lsrc])

    # Loop over Rp, Rm
    for plus in [True, False]:

        # Switches depending if plus or minus
        if plus:  # Starts from the bottom, upgoing
            pm = 1
            # layer_count = np.arange(depth.size - 2, minl - 1, -1)
            # izout = abs(lsrc - lrec)
            layer_count = np.arange(nlayer - 2, -1,
                                    -1)  # iterate over interfaces, so n_layer-1, n-layer - 2 for pythonic counting
            izout = nlayer - 1
            minmax = pm*maxl
        else:
            pm = -1
            # layer_count = np.arange(1, maxl + 1, 1)
            layer_count = np.arange(1, nlayer, 1)
            izout = 0
            minmax = pm*minl

        # If rec in last  and rec below src (plus) or
        # if rec in first and rec above src (minus), shift izout
        shiftplus = lrec < lsrc and lrec == 0 and not plus
        shiftminus = lrec > lsrc and lrec == depth.size-1 and plus
        if shiftplus or shiftminus:
            izout -= pm

        # Pre-allocate Ref and rloc
        # The number of the reflection coeff = the number of the interfaces : nlayer - 1
        Ref = np.zeros_like(Gam)  # Stores Ref for each layer with source and receiver and inbetween
        rloc = np.zeros_like(Gam[:, :, 0, :])  # Not all of them are needed, can be reduced (storing previous one)
        if ana_deriv:
            dRef = np.zeros(list(Gam.shape) + [nlayer],dtype=Gam.dtype)  # fifth dimension for derivative of dRef[i, ii, izout, iv] w.r.t. cond_n
            drloc = np.zeros_like(Gam[:, :, 0, :], dtype=Gam.dtype)  # within layer n w.r.t. cond of layer n
            drloc_pm = np.zeros_like(Gam[:, :, 0, :], dtype=Gam.dtype)  # within layer n w.r.t. cond of layer n + pm
            dRef_dRepm = np.zeros_like(Gam, dtype=Gam.dtype)

        # Calculate the reflection
        for idx, iz in enumerate(layer_count):
            # Eqs 65, A-12 - local reflection coefficients, read as if TM mode, lives within a layer iz
            for i in range(nfreq):
                ra = e_zH[i, iz + pm]  # (swapped to zeta for TE-mode)
                rb = e_zH[i, iz]
                for ii in range(noff):
                    for iv in range(nlambda):
                        rloca = ra*Gam[i, ii, iz, iv]
                        rlocb = rb*Gam[i, ii, iz + pm, iv]
                        rloc[i, ii, iv] = (rloca - rlocb)/(rloca + rlocb)
                        if ana_deriv:
                            # Derivative of rloc of layer iz w.r.t. conductivity of iz
                            # TODO: Fix mistake in document Expressions for the gradients wrt conductivity
                            # drloc[i, ii, iv] = ra * dGam[i, ii, iz, iv] * (1 - rloc[i, ii, iv]) + Gam[i, ii, iz + pm, iv] * (1 + rloc[i, ii, iv])
                            # drloc[i, ii, iv] /= (rloca + rlocb)
                            drloc[i, ii, iv] = (ra*dGam[i, ii, iz, iv] - Gam[i, ii, iz + pm, iv]) - rloc[
                                i, ii, iv]*(ra*dGam[i, ii, iz, iv] + Gam[i, ii, iz + pm, iv])
                            drloc[i, ii, iv] /= (rloca + rlocb)
                            # drloc[i, ii, iv] = (ra * dGam[i, ii, iz, iv] - Gam[i, ii, iz + pm, iv])/ (rloca + rlocb)
                            # drloc[i, ii, iv] += (-1*(ra * Gam[i, ii, iz, iv] - rb * Gam[i, ii, iz + pm, iv]) * (ra * dGam[i, ii, iz, iv] + Gam[i,ii,iz+pm,iv])) / (rloca + rlocb)**2
                            # Derivative of rloc of layer iz w.r.t. conductivity of iz + pm
                            drloc_pm[i, ii, iv] = -rb*dGam[i, ii, iz + pm, iv]*(1 + rloc[i, ii, iv]) + \
                                                  Gam[i, ii, iz, iv]*(1 - rloc[i, ii, iv])
                            drloc_pm[i, ii, iv] /= (rloca + rlocb)

            # In first layer tRef = rloc
            if iz == layer_count[0]:
                Ref[:, :, iz, :] = rloc[:, :, :].copy()
                if ana_deriv:
                    dRef[:, :, iz, :, iz] = drloc[:, :, :].copy()
                    dRef[:, :, iz, :, iz + pm] = drloc_pm[:, :, :].copy()

                    # TODO: (!!) What about the dREF in layer iz, but the derivative of the conductivity of layer iz+1?
                    # So in a 6-layered case, you have 5 global reflection coefficients, but 6 derivatives of the global reflection coefficients w.r.t. the conductivity of the layers
                    # So, I guess, we are forgetting one contribution, namely \partial R_{n} / \partial \sigma_{n+1}
                    # R_{n} explicitly depends on \sigma_{n+1} via rloc_{n}.
                    # I think, in the current implementation, we totally ignore the conductivity in the last layer, except via drloc_pm.
            else:
                ddepth = depth[iz + 1 + pm] - depth[iz + pm]

                # Eqs 64, A-11
                for i in range(nfreq):
                    for ii in range(noff):
                        for iv in range(nlambda):
                            term = Ref[i, ii, iz + pm, iv]*np.exp(
                                -2*Gam[i, ii, iz + pm, iv]*ddepth)
                            Ref[i, ii, iz, iv] = (rloc[i, ii, iv] + term)/(
                                    1 + rloc[i, ii, iv]*term)
                            if ana_deriv:
                                # Derivative of tRef of layer iz w.r.t. conductivity of iz
                                # Current dtRef is from previous layer, now recursively current tRef_prev and dtRef are for n + 1
                                E = np.exp(-2*Gam[i, ii, iz + pm, iv]*ddepth)
                                # dRef[:, :, iz, :, iz] = (1 - tRef[i, ii, iv] * tRef_prev[i, ii, iv] * E)
                                # dRef[:, :, iz, :, iz] /= (1 + rloc[i, ii, iv] * tRef_prev[i, ii, iv] * E)  # eq. 1.24 #TODO: move exponential out of the loop
                                a = 1 - Ref[i, ii, iz, iv]*Ref[i, ii, iz + pm, iv]*E
                                a /= (1 + rloc[i, ii, iv]*Ref[i, ii, iz + pm, iv]*E)
                                dRef[i, ii, iz, iv, iz] = a*drloc[i, ii, iv]  # eq. 1.23

                                # Derivative of tRef of layer iz w.r.t. conductivity of iz + pm
                                b = (1 - rloc[i, ii, iv]*Ref[i, ii, iz, iv])*E
                                b /= (1 + rloc[i, ii, iv]*Ref[i, ii, iz + pm, iv]*E)
                                dRef_dRepm[i, ii, iz, iv] = b.copy()

                                c = Ref[i, ii, iz + pm, iv]*(1 - rloc[i, ii, iv]*Ref[i, ii, iz, iv])
                                c /= (1 + rloc[i, ii, iv]*Ref[i, ii, iz + pm, iv]*E)

                                d = -2*ddepth*E*dGam[i, ii, iz, iv]

                                dRef[i, ii, iz, iv, iz + pm] = a*drloc_pm[i, ii, iv] + b*dRef[
                                    i, ii, iz + pm, iv, iz + pm] + c*d

                                # Derivative of tRef of layer iz w.r.t. conductivity of iz + n*pm
                                for n in range(1, idx + 1):  # recursively to iz + 2 and so on
                                    dRef[i, ii, iz, iv, iz + (n + 1)*pm] = dRef[
                                        i, ii, iz + n*pm, iv, iz + (n + 1)*pm]  # Derivative w.r.t conductivity
                                    for m in np.arange(0, n):  # Derivative w.r.t. other Ref's of other layers
                                        dRef[i, ii, iz, iv, iz + (n + 1)*pm] *= dRef_dRepm[i, ii, iz + m*pm, iv]

            # TODO: This is wat it used to be, now it returns all Ref's!
            # The global reflection coefficient is given back for all layers
            # between and including src- and rec-layer
            # Ref[:, :, iz, :] = tRef[:]
            # The global reflection coefficient JACOBIAN is given back for all layers
            # between bottom/top and including src- and rec-layer

        # If lsrc = lrec, we just store the last values
        # if lsrc == lrec and layer_count.size > 0:
        #    out = np.zeros_like(Ref[:, :, :1, :])
        #    out[:, :, 0, :] = tRef
        # else:
        #    out = Ref

        out = Ref

        # Store Ref in Rm/Rp
        if plus:
            Rm = out
            if ana_deriv:
                dRm = dRef
        else:
            Rp = out
            if ana_deriv:
                dRp = dRef
    # Return reflections (minus and plus)

    if debug:
        if ana_deriv:
            return Rm, Rp, dRm, dRp
        else:
            return Rm, Rp
    else:
        if ana_deriv:
            return Rm[:, :, minl:(maxl + 1), :], Rp[:, :, minl:(maxl + 1), :], dRm, dRp
        else:
            return Rm[:, :, minl:(maxl + 1), :], Rp[:, :, minl:(maxl + 1), :]


@nb.njit(**_numba_setting)
def fields(depth, Rp, Rm, Gam, lrec, lsrc, zsrc, ab, TM, ana_deriv: bool = False, dRp=None, dRm=None, dGam=None):
    r"""Calculate Pu+, Pu-, Pd+, Pd-.

    .. math::
        :label: fields

        P^{u\pm}_s, P^{d\pm}_s, \bar{P}^{u\pm}_s, \bar{P}^{d\pm}_s;
        P^{u\pm}_{s-1}, P^{u\pm}_n, \bar{P}^{u\pm}_{s-1}, \bar{P}^{u\pm}_n;
        P^{d\pm}_{s+1}, P^{d\pm}_n, \bar{P}^{d\pm}_{s+1}, \bar{P}^{d\pm}_n

    This function corresponds to equations 81/82, 95/96, 103/104, A-8/A-9,
    A-24/A-25, and A-32/A-33 in [HuTS15]_, and loosely to the corresponding
    files `Pdownmin.F90`, `Pdownplus.F90`, `Pupmin.F90`, and `Pdownmin.F90`.

    This function is called from the function :func:`greenfct`.

    """

    nfreq, noff, nlayer, nlambda = Gam[:, :, :, :].shape

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
    dRmp = dRm
    dRpm = dRp

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
            dPu = np.zeros(list(iGam.shape) + [nlayer], dtype=Gam.dtype)
            continue
        # No downgoing field if rec is in first layer or above src
        if not up and (lrec == 0 or lrec < lsrc):
            Pd = np.zeros_like(iGam)
            dPd = np.zeros(list(iGam.shape) + [nlayer], dtype=Gam.dtype)
            continue

        # Swaps if up=True
        if up:
            if not last_layer:
                dp, dm = dm, dp
            else:
                dp = dm
            Rmp, Rpm = Rpm, Rmp
            dRmp, dRpm = dRpm, dRmp
            first_layer, last_layer = last_layer, first_layer
            rsrcl = nlsr-1  # src-layer in refl. (Rp/Rm), last (nlsr-1) if up
            izrange = range(nlsr-2)
            isr = lrec
            last = 0
            pup = 1
            if not plus:
                mupm = -1

        P = np.zeros_like(iGam)
        dP = np.zeros(list(iGam.shape) + [nlayer], dtype=Gam.dtype)

        # Calculate Pu+, Pu-, Pd+, Pd-
        if lsrc == lrec:  # rec in src layer; Eqs  81/82, A-8/A-9
            if last_layer:  # If src/rec are in top (up) or bottom (down) layer
                """
                tRpm is zero, as there is no reflection from the bottom/top. 
                dRpm is zero 
                M = 1 
                """
                for i in range(nfreq):
                    for ii in range(noff):
                        for iv in range(nlambda):
                            tRmp = Rmp[i, ii, 0, iv]
                            tiGam = iGam[i, ii, iv]
                            E1 = np.exp(-tiGam*dm)
                            P[i, ii, iv] = tRmp*E1
                            if ana_deriv:
                                # Not all derivatives iterate over the number of layers, so 3dim
                                # Depths; dp and dm are swapped if up=True
                                # Rmp = Rm;  swapped if up=True
                                # Rpm = Rp;  swapped if up=True
                                # dm and dp swapped if up=True
                                t1 = E1
                                t7 = tRmp
                                for v in range(nlayer):
                                    # TODO: number of iterations may be reduced. Check the layers to iterate over
                                    if v == lsrc:
                                        dgam = dGam[i, ii, lsrc, iv]
                                    else:
                                        dgam = 0
                                    t8 = -dm*E1*dgam
                                    dP[i, ii, iv, v] = t1*dRmp[i, ii, lsrc, iv, v] + t7*t8

            else:           # If src and rec are in any layer in between
                for i in range(nfreq):
                    for ii in range(noff):
                        for iv in range(nlambda):
                            tiGam = iGam[i, ii, iv]
                            tRpm = Rpm[i, ii, 0, iv]  # TODO: check if Rpm indeed has only the R for one layer
                            tRmp = Rmp[i, ii, 0, iv]
                            E1 = np.exp(-tiGam*dm)
                            E2 = np.exp(-tiGam*(ds + dp))
                            E3 = np.exp(-2*tiGam*ds)
                            p2 = pm*tRpm*E2
                            M = 1 - tRmp*tRpm*E3
                            P[i, ii, iv] = (E1 + p2)*tRmp/M
                            if ana_deriv:
                                # Not all derivatives iterate over the number of layers, so 3dim
                                # Depths; dp and dm are swapped if up=True
                                # Rmp = Rm;  swapped if up=True
                                # Rpm = Rp;  swapped if up=True
                                # dm and dp swapped if up=True
                                t1 = (E1 + p2)*1/M
                                t3 = pm*tRmp/M*E2
                                t5 = P[i, ii, iv]/M  # TODO: Check, does this term requires an additional *-1?
                                t7 = tRmp/M
                                t9 = pm*tRpm*tRmp/M

                                for v in range(nlayer):  # TODO: number of iterations may be reduced. Check the layers to iterate over
                                    if v == lsrc:
                                        dgam = dGam[i, ii, v, iv]
                                    else:
                                        dgam = 0
                                    t6 = E3*(tRmp*dRpm[i, ii, lsrc, iv, v] + tRpm*dRmp[
                                        i, ii, lsrc, iv, v] - 2*tRpm*tRmp*ds*dgam)
                                    t8 = -dm*E1*dgam
                                    t10 = -(ds + dp)*E2*dgam
                                    dP[i, ii, iv, v] = t1*dRmp[i, ii, lsrc, iv, v] + t3*dRpm[
                                        i, ii, lsrc, iv, v] + t5*t6 + t7*t8 + t9*t10


        else:           # rec above (up) / below (down) src layer
            #           # Eqs  95/96,  A-24/A-25 for rec above src layer
            #           # Eqs 103/104, A-32/A-33 for rec below src layer

            # First compute P_{s-1} (up) / P_{s+1} (down)
            iRpm = Rpm[:, :, rsrcl, :]
            if first_layer:  # If src is in bottom (up) / top (down) layer
                for i in range(nfreq):
                    for ii in range(noff):
                        for iv in range(nlambda):
                            tiRpm = iRpm[i, ii, iv]
                            tiGam = iGam[i, ii, iv]
                            P[i, ii, iv] = (1 + tiRpm)*mupm*np.exp(-tiGam*dp)
                            if ana_deriv:
                                raise NotImplementedError("Case: first layer")
            else:
                for i in range(nfreq):
                    for ii in range(noff):
                        for iv in range(nlambda):
                            iRmp = Rmp[i, ii, rsrcl, iv]
                            tiGam = iGam[i, ii, iv]
                            tRpm = iRpm[i, ii, iv]
                            p1 = mupm*np.exp(-tiGam*dp)
                            p2 = pm*mupm*iRmp*np.exp(-tiGam * (ds+dm))
                            p3 = (1 + tRpm)/(1 - iRmp*tRpm*np.exp(-2*tiGam*ds))
                            P[i, ii, iv] = (p1 + p2)*p3
                            if ana_deriv:
                                raise NotImplementedError("Case: rec is not in src layer")

            # If up or down and src is in last but one layer
            if up or (not up and lsrc+1 < depth.size-1):
                ddepth = depth[lsrc+1-1*pup]-depth[lsrc-1*pup]
                for i in range(nfreq):
                    for ii in range(noff):
                        for iv in range(nlambda):
                            tiRpm = Rpm[i, ii, rsrcl-1*pup, iv]
                            tiGam = Gam[i, ii, lsrc-1*pup, iv]
                            P[i, ii, iv] /= 1 + tiRpm*np.exp(-2*tiGam*ddepth)
                            if ana_deriv:
                                raise NotImplementedError("Case: src is in last but one layer")

            # Second compute P for all other layers
            if nlsr > 2:
                for iz in izrange:
                    ddepth = depth[isr+iz+pup+1]-depth[isr+iz+pup]
                    for i in range(nfreq):
                        for ii in range(noff):
                            for iv in range(nlambda):
                                tiRpm = Rpm[i, ii, iz+pup, iv]
                                piGam = Gam[i, ii, isr+iz+pup, iv]
                                p1 = (1+tiRpm)*np.exp(-piGam*ddepth)
                                P[i, ii, iv] *= p1
                                if ana_deriv:
                                    raise NotImplementedError("Case: number of layers between src and rec > 2")

                    # If rec/src NOT in first/last layer (up/down)
                    if isr+iz != last:
                        ddepth = depth[isr+iz+1] - depth[isr+iz]
                        for i in range(nfreq):
                            for ii in range(noff):
                                for iv in range(nlambda):
                                    tiRpm = Rpm[i, ii, iz, iv]
                                    piGam2 = Gam[i, ii, isr+iz, iv]
                                    p1 = 1 + tiRpm*np.exp(-2*piGam2 * ddepth)
                                    P[i, ii, iv] /= p1
                                    if ana_deriv:
                                        raise NotImplementedError("Case: If rec/src NOT in first/last layer (up/down)")

        # Store P in Pu/Pd
        if up:
            Pu = P
            if ana_deriv:
                dPu = dP
        else:
            Pd = P
            if ana_deriv:
                dPd = dP

    # Return fields (up- and downgoing)
    if ana_deriv:
        return Pu, Pd, dPu, dPd
    else:
        return Pu, Pd


# Angle Factor

def angle_factor(angle, ab, msrc, mrec):
    r"""Return the angle-dependent factor.

    The whole calculation in the wavenumber domain is only a function of the
    distance between the source and the receiver, it is independent of the
    angel. The angle-dependency is this factor, which can be applied to the
    corresponding parts in the wavenumber or in the frequency domain.

    The :func:`angle_factor` corresponds to the sine and cosine-functions in
    Eqs 105-107, 111-116, 119-121, 123-128.

    This function is called from one of the Hankel functions in
    :mod:`empymod.transform`.  Consult the modelling routines in
    :mod:`empymod.model` for a description of the input and output parameters.

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
    ang_fact = fct(eval_angle)

    # Ensure cos([pi/2, 3pi/2]) and sin([pi, 2pi]) are zero (floating pt issue)
    ang_fact[np.isclose(np.abs(eval_angle), test_ang_1, 1e-10, 1e-14)] = 0
    ang_fact[np.isclose(np.abs(eval_angle), test_ang_2, 1e-10, 1e-14)] = 0

    return ang_fact


# Analytical solutions

@np.errstate(all='ignore')
def fullspace(off, angle, zsrc, zrec, etaH, etaV, zetaH, zetaV, ab, msrc,
              mrec):
    r"""Analytical full-space solutions in the frequency domain.

    .. math::
        :label: fullspace

        \hat{G}^{ee}_{\alpha\beta}, \hat{G}^{ee}_{3\alpha},
        \hat{G}^{ee}_{33}, \hat{G}^{em}_{\alpha\beta}, \hat{G}^{em}_{\alpha 3}

    This function corresponds to equations 45--50 in [HuTS15]_, and loosely to
    the corresponding files `Gin11.F90`, `Gin12.F90`, `Gin13.F90`, `Gin22.F90`,
    `Gin23.F90`, `Gin31.F90`, `Gin32.F90`, `Gin33.F90`, `Gin41.F90`,
    `Gin42.F90`, `Gin43.F90`, `Gin51.F90`, `Gin52.F90`, `Gin53.F90`,
    `Gin61.F90`, and `Gin62.F90`.

    This function is called from one of the modelling routines in
    :mod:`empymod.model`. Consult these modelling routines for a description of
    the input and output parameters.

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
        if ab in [15, 25]:
            coo1, coo3 = coo3, coo1
            coo2, coo4 = coo4, coo2

        # 24/25: Swap src/rec
        if ab in [24, 25]:
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


@np.errstate(all='ignore')
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
    diffusive approximation. See `solution`-parameter.

    This function is called from one of the modelling routines in
    :mod:`empymod.model`. Consult these modelling routines for a description of
    the input and solution parameters.

    """
    from scipy import special  # Lazy for faster CLI load

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
        dtype = np.float64

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
        f0p = sp.special.erfc(np.sqrt(tp/time))
        f0m = sp.special.erfc(np.sqrt(tm/time))
        fs0p = sp.special.erfc(np.sqrt(tsp/time))
        fs0m = sp.special.erfc(np.sqrt(tsm/time))

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
        izr = rh == 0  # index where rh = 0
        iir = np.invert(izr)  # invert of izr
        izh = hm == 0  # index where hm = 0
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
            return np.exp(-np.real(gamH)*hp)*sp.special.ive(nr, xim)

        def BK(xip, nr):
            r"""Return BK_nr."""
            if np.isrealobj(xip):
                # To keep it real in Laplace-domain [exp(-1j*0) = 1-0j].
                return sp.special.kve(nr, xip)
            else:
                return np.exp(-1j*np.imag(xip))*sp.special.kve(nr, xip)

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
            def coeff_dk(k, K):
                r"""Return coefficients Dk for k, K."""
                n = np.arange((k+1)//2, min([k, K/2])+.5, 1)
                Dk = n**(K/2)*sp.special.factorial(2*n)/sp.special.factorial(n)
                Dk /= sp.special.factorial(n-1)*sp.special.factorial(k-n)
                Dk /= sp.special.factorial(2*n-k)*sp.special.factorial(K/2-n)
                return Dk.sum()*(-1)**(k+K/2)

            for k in range(1, K+1):
                sval = k*np.log(2)/time
                cair = airwave(sval, hp, rp, res, fab, delta)
                air += coeff_dk(k, K)*cair.real/k

        else:  # Time-domain impulse response
            thp = mu_0*hp**2/(4*res)
            trh = mu_0*rh**2/(8*res)
            P1 = (mu_0**2*hp*np.exp(-thp/time))/(res*32*np.pi*time**3)
            P2 = 2*(delta - (x*y)/rh**2)*sp.special.ive(1, trh/time)
            P3 = mu_0/(2*res*time)*(rh**2*delta - x*y)-delta
            P4 = sp.special.ive(0, trh/time) - sp.special.ive(1, trh/time)

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
