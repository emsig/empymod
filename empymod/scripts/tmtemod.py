r"""
:mod:`tmtemod` -- Calculate up- and down-going TM and TE modes
==============================================================

This add-on for ``empymod`` adjusts [HuTS15]_ for TM/TE-split. The development
was initiated by the development of
https://github.com/empymod/csem-ziolkowski-and-slob ([ZiSl19]_).

This is a stripped-down version of ``empymod`` with a lot of simplifications
but an important addition. The modeller ``empymod`` returns the total field,
hence not distinguishing between TM and TE mode, and even less between up- and
down-going fields. The reason behind this is simple: The derivation of
[HuTS15]_, on which ``empymod`` is based, returns the total field. In this
derivation each mode (TM and TE) contains non-physical contributions. The
non-physical contributions have opposite signs in TM and TE, so they cancel
each other out in the total field. However, in order to obtain the correct TM
and TE contributions one has to remove these non-physical parts.

This is what this routine does, but only for an x-directed electric source with
an x-directed electric receiver, and in the frequency domain (src and rec in
same layer). This version of ``dipole`` returns the signal separated into TM++,
TM+-, TM-+, TM--, TE++, TE+-, TE-+, and TE-- as well as the direct field TM and
TE contributions. The first superscript denotes the direction in which the
field diffuses towards the receiver and the second superscript denotes the
direction in which the field diffuses away from the source. For both the
plus-sign indicates the field diffuses in the downward direction and the
minus-sign indicates the field diffuses in the upward direction. It uses
``empymod`` wherever possible. See the corresponding functions in ``empymod``
for more explanation and documentation regarding input parameters. There are
important limitations:

- ``ab`` == 11                   [=> x-directed el. source & el. receivers]
- ``signal`` == None             [=> only frequency domain]
- ``xdirect`` == False           [=> direct field calc. in wavenr-domain]
- ``ht`` == 'fht'
- ``htarg`` == 'key_201_2012'
- Options ``ft``, ``ftarg``, ``opt``, and ``loop`` are not available.
- ``lsrc`` == ``lrec``           [=> src & rec are assumed in same layer!]
- Model must have more than 1 layer
- Electric permittivity and magnetic permeability are isotropic.
- Only one frequency at once.


Theory
------

The derivation of [HuTS15]_, on which ``empymod`` is based, returns the total
field. Internally it also calculates TM and TE modes, and sums these up.
However, the separation into TM and TE mode introduces a singularity at
:math:`\kappa = 0`. It has no contribution in the space-frequency domain to the
total fields, but it introduces non-physical events in each mode with opposite
sign (so they cancel each other out in the total field). In order to obtain the
correct TM and TE contributions one has to remove these non-physical parts.

To remove the non-physical part we use the file ``tmtemod.py`` in this
directory. This routine is basically a heavily simplified version of
``empymod`` with the following limitations outlined above.

So ``tmtemod.py`` returns the signal separated into TM++, TM+-, TM-+, TM--,
TE++, TE+-, TE-+, and TE-- as well as the direct field TM and TE contributions.
The first superscript denotes the direction in which the field diffuses towards
the receiver and the second superscript denotes the direction in which the
field diffuses away from the source. For both the plus-sign indicates the field
diffuses in the downward direction and the minus-sign indicates the field
diffuses in the upward direction. The routine uses ``empymod`` wherever
possible, see the corresponding functions in ``empymod`` for more explanation
and documentation regarding input parameters.

Please note that the notation in [HuTS15]_ differs from the notation in
[ZiSl19]_. I specify therefore always, which notification applies, either
*Hun15* or *Zio19*.

We start with equation (105) in *Hun15*:

.. math::

    \hat{G}^{ee}_{xx}(\boldsymbol{x}, \boldsymbol{x'}, \omega) =
    \hat{G}^{ee;i}_{xx;s}(\boldsymbol{x}-\boldsymbol{x'}, \omega)
    + \frac{1}{8\pi}\int^\infty_{\kappa=0}
    \left(\frac{\Gamma_s \tilde{g}^{tm}_{hh;s}}{\eta_s} -
    \frac{\zeta_s \tilde{g}^{te}_{zz;s}}{\bar{\Gamma}_s}\right)
    J_0(\kappa r)\kappa d \kappa

.. math::

    - \frac{\cos(2\phi)}{8\pi}\int^\infty_{\kappa=0}
    \left(\frac{\Gamma_s \tilde{g}^{tm}_{hh;s}}{\eta_s} +
    \frac{\zeta_s \tilde{g}^{te}_{zz;s}}{\bar{\Gamma}_s}\right)
    J_2(\kappa r)\kappa d \kappa .

Ignoring the incident field, and using
:math:`J_2 = \frac{2}{\kappa r}J_1 - J_0` to avoid
:math:`J_2`-integrals, we get

.. math::

    \hat{G}^{ee}_{xx}(\boldsymbol{x}, \boldsymbol{x'}, \omega) =
    \frac{1}{8\pi}\int^\infty_{\kappa=0}
    \left(\frac{\Gamma_s \tilde{g}^{tm}_{hh;s}}{\eta_s}-
    \frac{\zeta_s \tilde{g}^{te}_{zz;s}}{\bar{\Gamma}_s}\right)
    J_0(\kappa r)\kappa\,{\mathrm{d}}\kappa

.. math::

    + \frac{\cos(2\phi)}{8\pi}\int^\infty_{\kappa=0}
    \left(\frac{\Gamma_s \tilde{g}^{tm}_{hh;s}}{\eta_s} +
    \frac{\zeta_s \tilde{g}^{te}_{zz;s}}{\bar{\Gamma}_s}\right)
    J_0(\kappa r)\kappa\,{\mathrm{d}}\kappa

.. math::

    - \frac{\cos(2\phi)}{4\pi r}\int^\infty_{\kappa=0}
    \left(\frac{\Gamma_s \tilde{g}^{tm}_{hh;s}}{\eta_s} +
    \frac{\zeta_s \tilde{g}^{te}_{zz;s}}{\bar{\Gamma}_s}\right)
    J_1(\kappa r)\,{\mathrm{d}}\kappa .

From this the TM- and TE-parts follow as

.. math::

     {\mathrm{TE}} = \frac{\cos(2\phi)-1}{8\pi}\int^\infty_{\kappa=0}
     \frac{\zeta_s \tilde{g}^{te}_{zz;s}}{\bar{\Gamma}_s}
     J_0(\kappa r)\kappa\,{\mathrm{d}}\kappa
      - \frac{\cos(2\phi)}{4\pi r}\int^\infty_{\kappa=0}
     \frac{\zeta_s \tilde{g}^{te}_{zz;s}}{\bar{\Gamma}_s}
     J_1(\kappa r)\,{\mathrm{d}}\kappa ,

.. math::

       {\mathrm{TM}} = \frac{\cos(2\phi)+1}{8\pi}\int^\infty_{\kappa=0}
     \frac{\Gamma_s \tilde{g}^{tm}_{hh;s}}{\eta_s}
     J_0(\kappa r)\kappa\,{\mathrm{d}}\kappa
     - \frac{\cos(2\phi)}{4\pi r}\int^\infty_{\kappa=0}
     \frac{\Gamma_s \tilde{g}^{tm}_{hh;s}}{\eta_s}
     J_1(\kappa r)\,{\mathrm{d}}\kappa .

Equations (108) and (109) in Hun15 yield the required parameters
:math:`\tilde{g}^{tm}_{hh;s}` and :math:`\tilde{g}^{te}_{zz;s}`,

.. math::

     \tilde{g}^{tm}_{hh;s} = P^{u-}_s W^u_s + P^{d-}_s W^d_s ,

.. math::

     \tilde{g}^{te}_{zz;s} = \bar{P}^{u+}_s \bar{W}^u_s +
                              \bar{P}^{d+}_s \bar{W}^d_s \ .

The parameters :math:`P^{u\pm}_s` and :math:`P^{d\pm}_s` are given in equations
(81) and (82), :math:`\bar{P}^{u\pm}_s` and :math:`\bar{P}^{d\pm}_s` in
equations (A-8) and (A-9); :math:`W^u_s` and :math:`W^d_s` in equation (74)
in Hun15. This yields

.. math::

     \tilde{g}^{te}_{zz;s} =
     \frac{\bar{R}_s^+}{\bar{M}_s}\left\{\exp[-\bar{\Gamma}_s(z_s-z+d^+)] +
     \bar{R}_s^-\exp[-\bar{\Gamma}_s(z_s-z+d_s+d^-)]\right\}

.. math::

     + \frac{\bar{R}_s^-}{\bar{M}_s}
     \left\{\exp[-\bar{\Gamma}_s(z-z_{s-1}+d^-)]+
     \bar{R}_s^+\exp[-\bar{\Gamma}_s(z-z_{s-1}+d_s+d^+)]\right\} ,

.. math::

     =\frac{\bar{R}_s^+}{\bar{M}_s}\left\{\exp[-\bar{\Gamma}_s(2z_s-z-z')]
     + \bar{R}_s^-\exp[-\bar{\Gamma}_s(z'-z+2d_s)]\right\}

.. math::

     + \frac{\bar{R}_s^-}{\bar{M}_s}
     \left\{\exp[-\bar{\Gamma}_s(z+z'-2z_{s-1})]+
     \bar{R}_s^+\exp[-\bar{\Gamma}_s(z-z'+2d_s)]\right\} ,


where :math:`d^\pm` is taken from the text below equation (67). There are four
terms in the right-hand side, two in the first line and two in the second line.
The first term in the first line is the integrand of TE+-, the second term in
the first line corresponds to TE++, the first term in the second line is TE-+,
and the second term in the second line is TE--.

If we look at TE+-, we have

.. math::

   \tilde{g}^{te+-}_{zz;s} =
   \frac{\bar{R}_s^+}{\bar{M}_s}\exp[-\bar{\Gamma}_s(2z_s-z-z')] \ ,


and therefore

.. math::

   {\mathrm{TE}}^{+-} = \frac{\cos(2\phi)-1}{8\pi}\int^\infty_{\kappa=0}
   \frac{\zeta_s \bar{R}_s^+}{\bar{\Gamma}_s\bar{M}_s}
   \exp[-\bar{\Gamma}_s(2z_s-z-z')]
   J_0(\kappa r)\kappa\,{\mathrm{d}}\kappa

.. math::
   - \frac{\cos(2\phi)}{4\pi r}\int^\infty_{\kappa=0}
   \frac{\zeta_s \bar{R}_s^+}{\bar{\Gamma}_s\bar{M}_s}
   \exp[-\bar{\Gamma}_s(2z_s-z-z')]
   J_1(\kappa r)\,{\mathrm{d}}\kappa .

We can compare this to equation (4.165) in Zio19, with :math:`\hat{I}^e_x=1`
and slightly re-arranging it to look more alike, we get

.. math::

   \hat{E}^{+-}_{xx;H} = \frac{y^2}{4\pi r^2}
   \int^\infty_{\kappa=0} \frac{\zeta_1}{\Gamma_1}
   \frac{R^-_{H;1}}{M_{H;1}}
   \exp(-\Gamma_1 h^{+-})J_0(\kappa r)\kappa d\kappa

.. math::

  + \frac{x^2-y^2}{4\pi r^3}
  \int^\infty_{\kappa=0} \frac{\zeta_1}{\Gamma_1}
  \left(\frac{R^-_{H;1}}{M_{H;1}} -
  \frac{R^-_{H;1}(\kappa=0)}{M_{H;1}(\kappa=0)}\right)
  \exp(-\Gamma_1 h^{+-})J_1(\kappa r) d\kappa


.. math::

   - \frac{\zeta_1 (x^2-y^2)}{4\pi\gamma_1 r^4}
   \frac{R^-_{H;1}(\kappa=0)}{M_{H;1}(\kappa=0)}
   \exp(-\gamma_1 R^{+-}) .

The notation in this equation follows Zio19.

The difference between the two previous equations is that the first one
contains non-physical contributions. These have opposite signs in TM+- and
TE+-, and therefore cancel each other out. But if we want to know the specific
contributions from TM and TE we have to remove them. The non-physical
contributions only affect the :math:`J_1`-integrals, and only for :math:`\kappa
= 0`.

The following lists for all 8 cases the term that has to be removed, in the
notation of Zio19 (for the notation as in Hun15 see the implementation in
``tmtemod.py``):

.. math::

  TE^{++} = + \frac{\zeta_1 (x^2-y^2)}{4\pi\gamma_1 r^4}
  \frac{\exp(-\gamma_1 |h^-|) }{M_{H;1}(\kappa=0)} ,

.. math::

  TE^{-+} = - \frac{\zeta_1 (x^2-y^2)}{4\pi\gamma_1 r^4}
  \frac{R^+_{H;1}(\kappa=0)\exp(-\gamma_1 h^{-+}) }{M_{H;1}(\kappa=0)} ,

.. math::

  TE^{+-} = - \frac{\zeta_1 (x^2-y^2)}{4\pi\gamma_1 r^4}
  \frac{R^-_{H;1}(\kappa=0)\exp(-\gamma_1 h^{+-}) }{M_{H;1}(\kappa=0)} ,

.. math::

  TE^{--} = + \frac{\zeta_1 (x^2-y^2)}{4\pi\gamma_1 r^4}
  \frac{R^+_{H;1}(\kappa=0)R^-_{H;1}(\kappa=0)\exp(-\gamma_1 h^{--}) }
  {M_{H;1}(\kappa=0)} ,

.. math::

  TM^{++} = - \frac{\zeta_1 (x^2-y^2)}{4\pi\gamma_1 r^4}
  \frac{\exp(-\gamma_1 |h^-|) }{M_{V;1}(\kappa=0)} ,

.. math::

  TM^{-+} = - \frac{\zeta_1 (x^2-y^2)}{4\pi\gamma_1 r^4}
  \frac{R^+_{V;1}(\kappa=0)\exp(-\gamma_1 h^{-+}) }{M_{V;1}(\kappa=0)} ,

.. math::

  TM^{+-} = - \frac{\zeta_1 (x^2-y^2)}{4\pi\gamma_1 r^4}
  \frac{R^-_{V;1}(\kappa=0)\exp(-\gamma_1 h^{+-}) }{M_{V;1}(\kappa=0)} ,

.. math::

  TM^{--} = - \frac{\zeta_1 (x^2-y^2)}{4\pi\gamma_1 r^4}
  \frac{R^+_{V;1}(\kappa=0)R^-_{V;1}(\kappa=0)\exp(-\gamma_1 h^{--}) }
  {M_{V;1}(\kappa=0)} .



Note that in the first and fourth equations the correction terms have opposite
sign as those in the fifth and eighth equations because at :math:`\kappa=0` the
TM and TE mode correction terms are equal. Also note that in the second and
third equations the correction terms have the same sign as those in the sixth
and seventh equations because at :math:`\kappa=0` the TM and TE mode reflection
responses in those terms are equal but with opposite sign:
:math:`R^\pm_{V;1}(\kappa=0) = -R^\pm_{V;1}(\kappa=0)`.

Hun15 uses :math:`\phi`, whereas Zio19 uses :math:`x`, :math:`y`, for which we
can use

.. math::

   \cos(2\phi) = -\frac{x^2-y^2}{r^2} \ .


"""
# Copyright 2017-2019 The empymod Developers.
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

from empymod.filters import key_201_2012
from empymod.kernel import reflections, angle_factor
from empymod.utils import (check_model, check_frequency, check_dipole, _strvar,
                           get_off_ang, get_layer_nr, printstartfinish)

__all__ = ['dipole']


def dipole(src, rec, depth, res, freqtime, aniso=None, eperm=None, mperm=None,
           verb=2):
    r"""Return the electromagnetic field due to a dipole source.

    This is a modified version of ``empymod.model.dipole()``. It returns the
    separated contributions of TM--, TM-+, TM+-, TM++, TMdirect, TE--, TE-+,
    TE+-, TE++, and TEdirect.

    Parameters
    ----------
    src, rec : list of floats or arrays
        Source and receiver coordinates (m): [x, y, z].
        The x- and y-coordinates can be arrays, z is a single value.
        The x- and y-coordinates must have the same dimension.

        Sources or receivers placed on a layer interface are considered in the
        upper layer.

        Sources and receivers must be in the same layer.

    depth : list
        Absolute layer interfaces z (m); #depth = #res - 1
        (excluding +/- infinity).

    res : array_like
        Horizontal resistivities rho_h (Ohm.m); #res = #depth + 1.

    freqtime : float
        Frequency f (Hz). (The name ``freqtime`` is kept for consistency with
        ``empymod.model.dipole()``. Only one frequency at once.

    aniso : array_like, optional
        Anisotropies lambda = sqrt(rho_v/rho_h) (-); #aniso = #res.
        Defaults to ones.

    eperm : array_like, optional
        Relative electric permittivities epsilon (-);
        #eperm = #res. Default is ones.

    mperm : array_like, optional
        Relative magnetic permeabilities mu (-);
        #mperm = #res. Default is ones.

    verb : {0, 1, 2, 3, 4}, optional
        Level of verbosity, default is 2:
            - 0: Print nothing.
            - 1: Print warnings.
            - 2: Print additional runtime and kernel calls
            - 3: Print additional start/stop, condensed parameter information.
            - 4: Print additional full parameter information


    Returns
    -------
    TM, TE : list of ndarrays, (nfreq, nrec, nsrc)
        Frequency-domain EM field [V/m], separated into
        TM = [TM--, TM-+, TM+-, TM++, TMdirect]
        and
        TE = [TE--, TE-+, TE+-, TE++, TEdirect].

        However, source and receiver are normalised. So the source strength is
        1 A and its length is 1 m. Therefore the electric field could also be
        written as [V/(A.m2)].

        The shape of EM is (nfreq, nrec, nsrc). However, single dimensions
        are removed.

    """

    # === 1. LET'S START ============
    t0 = printstartfinish(verb)

    # === 2. CHECK INPUT ============
    # Check layer parameters
    model = check_model(depth, res, aniso, eperm, eperm, mperm, mperm, False,
                        verb)
    depth, res, aniso, epermH, epermV, mpermH, mpermV, _ = model

    # Check frequency => get etaH, etaV, zetaH, and zetaV
    frequency = check_frequency(freqtime, res, aniso, epermH, epermV, mpermH,
                                mpermV, verb)
    freq, etaH, etaV, zetaH, zetaV = frequency

    # Check src and rec
    src, nsrc = check_dipole(src, 'src', verb)
    rec, nrec = check_dipole(rec, 'rec', verb)

    # Get offsets
    off, ang = get_off_ang(src, rec, nsrc, nrec, verb)

    # Get layer number in which src and rec reside (lsrc/lrec)
    lsrc, zsrc = get_layer_nr(src, depth)
    lrec, zrec = get_layer_nr(rec, depth)

    # Check limitations of this routine compared to the standard ``dipole``
    if lsrc != lrec:                           # src and rec in same layer
        print("* ERROR   :: src and rec must be in the same layer; " +
              "<lsrc>/<lrec> provided: "+str(lsrc)+"/"+str(lrec))
        raise ValueError('src-z/rec-z')

    if depth.size < 2:                         # at least two layers
        print("* ERROR   :: model must have more than one layer; " +
              "<depth> provided: "+_strvar(depth[1:]))
        raise ValueError('depth')

    if freq.size > 1:                          # only 1 frequency
        print("* ERROR   :: only one frequency permitted; " +
              "<freqtime> provided: "+_strvar(freqtime))
        raise ValueError('frequency')

    # === 3. EM-FIELD CALCULATION ============
    # This part is a simplification of:
    # - model.fem()
    # - transform.dlf()
    # - kernel.wavenumber()

    # DLF filter we use
    filt = key_201_2012()

    # 3.1. COMPUTE REQUIRED LAMBDAS for given hankel-filter-base
    lambd = filt.base/off[:, None]

    # 3.2. CALL THE KERNEL
    PTM, PTE = greenfct(zsrc, zrec, lsrc, lrec, depth, etaH, etaV, zetaH,
                        zetaV, lambd)

    # 3.3. CARRY OUT THE HANKEL TRANSFORM WITH DLF
    factAng = angle_factor(ang, 11, False, False)
    zmfactAng = (factAng[:, np.newaxis]-1)/2
    zpfactAng = (factAng[:, np.newaxis]+1)/2
    fact = 4*np.pi*off

    # TE [uu, ud, du, dd, df]
    for i, val in enumerate(PTE):
        PTE[i] = (factAng*np.dot(-val, filt.j1)/off +
                  np.dot(zmfactAng*val*lambd, filt.j0))/fact

    # TM [uu, ud, du, dd, df]
    for i, val in enumerate(PTM):
        PTM[i] = (factAng*np.dot(-val, filt.j1)/off +
                  np.dot(zpfactAng*val*lambd, filt.j0))/fact

    # 3.4. Remove non-physical contributions

    # (Note: The T*dd corrections differ slightly from the equations given in
    # the accompanying pdf, due to the way the direct field is accounted for
    # in the book.)

    # General parameters
    Gam = np.sqrt((zetaH*etaH)[:, None, :, None])  # Gam for lambd=0
    iGam = Gam[:, :, lsrc, 0]
    lgam = np.sqrt(zetaH[:, lsrc]*etaH[:, lsrc])
    ddepth = np.r_[depth, np.inf]
    ds = ddepth[lsrc+1] - ddepth[lsrc]

    def get_rp_rm(z_eta):
        r"""Return Rp, Rm."""

        # Get Rp/Rm for lambd=0
        Rp, Rm = reflections(depth, z_eta, Gam, lrec, lsrc, False)

        # Depending on model Rp/Rm have 3 or 4 dimensions. Last two are
        # wavenumbers and layers btw src and rec, which both are 1.
        if Rp.ndim == 4:
            Rp = np.squeeze(Rp, axis=3)
        if Rm.ndim == 4:
            Rm = np.squeeze(Rm, axis=3)
        Rp = np.squeeze(Rp, axis=2)
        Rm = np.squeeze(Rm, axis=2)

        # Calculate reverberation M and general factor npfct
        Ms = 1 - Rp*Rm*np.exp(-2*iGam*ds)
        npfct = factAng*zetaH[:, lsrc]/(fact*off*lgam*Ms)

        return Rp, Rm, npfct

    # TE modes TE[uu, ud, du, dd]
    Rp, Rm, npfct = get_rp_rm(zetaH)

    PTE[0] += npfct*Rp*Rm*np.exp(-lgam*(2*ds - zrec + zsrc))
    PTE[1] += npfct*Rp*np.exp(-lgam*(2*ddepth[lrec+1] - zrec - zsrc))
    PTE[2] += npfct*Rm*np.exp(-lgam*(zrec + zsrc))
    PTE[3] += npfct*Rp*Rm*np.exp(-lgam*(2*ds + zrec - zsrc))

    # TM modes TM[uu, ud, du, dd]
    Rp, Rm, npfct = get_rp_rm(etaH)

    PTM[0] -= npfct*Rp*Rm*np.exp(-lgam*(2*ds - zrec + zsrc))
    PTM[1] += npfct*Rp*np.exp(-lgam*(2*ddepth[lrec+1] - zrec - zsrc))
    PTM[2] += npfct*Rm*np.exp(-lgam*(zrec + zsrc))
    PTM[3] -= npfct*Rp*Rm*np.exp(-lgam*(2*ds + zrec - zsrc))

    # 3.5 Reshape for number of sources
    for i, val in enumerate(PTE):
        PTE[i] = np.squeeze(val.reshape((-1, nrec, nsrc), order='F'))

    for i, val in enumerate(PTM):
        PTM[i] = np.squeeze(val.reshape((-1, nrec, nsrc), order='F'))

    # === 4. FINISHED ============
    printstartfinish(verb, t0)

    # return [TMuu, TMud, TMdu, TMdd, TMdf], [TEuu, TEud, TEdu, TEdd, TEdf]
    return PTM, PTE


def greenfct(zsrc, zrec, lsrc, lrec, depth, etaH, etaV, zetaH, zetaV, lambd):
    r"""Calculate Green's function for TM and TE.

    This is a modified version of empymod.kernel.greenfct(). See the original
    version for more information.

    """
    # GTM/GTE have shape (frequency, offset, lambda).
    # gamTM/gamTE have shape (frequency, offset, layer, lambda):

    for TM in [True, False]:

        # Define eta/zeta depending if TM or TE
        if TM:
            e_zH, e_zV, z_eH = etaH, etaV, zetaH   # TM: zetaV not used
        else:
            e_zH, e_zV, z_eH = zetaH, zetaV, etaH  # TE: etaV not used

        # Uppercase gamma
        Gam = np.sqrt((e_zH/e_zV)[:, None, :, None] *
                      (lambd*lambd)[None, :, None, :] +
                      (z_eH*e_zH)[:, None, :, None])

        # Gamma in receiver layer
        lrecGam = Gam[:, :, lrec, :]

        # Reflection (coming from below (Rp) and above (Rm) rec)
        Rp, Rm = reflections(depth, e_zH, Gam, lrec, lsrc, False)

        # Field propagators
        # (Up- (Wu) and downgoing (Wd), in rec layer); Eq 74
        if lrec != depth.size-1:  # No upgoing field prop. if rec in last
            ddepth = depth[lrec + 1] - zrec
            Wu = np.exp(-lrecGam*ddepth)
        else:
            Wu = np.full_like(lrecGam, 0+0j)
        if lrec != 0:     # No downgoing field propagator if rec in first
            ddepth = zrec - depth[lrec]
            Wd = np.exp(-lrecGam*ddepth)
        else:
            Wd = np.full_like(lrecGam, 0+0j)

        # Field at rec level (coming from below (Pu) and above (Pd) rec)
        Puu, Pud, Pdu, Pdd = fields(depth, Rp, Rm, Gam, lrec, lsrc, zsrc, TM)

        # Store in corresponding variable PT* = [T*uu, T*ud, T*du, T*dd]
        df = np.exp(-lrecGam*abs(zsrc - zrec))  # direct field
        fTM = Gam[:, :, lrec, :]/etaH[:, None, lrec, None]
        fTE = zetaH[:, None, lsrc, None]/Gam[:, :, lsrc, :]
        if TM:
            PTM = [Puu*Wu*fTM, Pud*Wu*fTM, Pdu*Wd*fTM, Pdd*Wd*fTM, -df*fTM]
        else:
            PTE = [Puu*Wu*fTE, Pud*Wu*fTE, Pdu*Wd*fTE, Pdd*Wd*fTE, df*fTE]

    # Return Green's functions
    return PTM, PTE


def fields(depth, Rp, Rm, Gam, lrec, lsrc, zsrc, TM):
    r"""Calculate Pu+, Pu-, Pd+, Pd-.

    This is a modified version of empymod.kernel.fields(). See the original
    version for more information.

    """
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
    if TM:
        plus = False
    else:
        plus = True

    # Sign-switches
    pm = 1     # + if plus=True, - if plus=False
    if not plus:
        pm = -1

    # Calculate down- and up-going fields
    for up in [False, True]:

        # No upgoing field if rec is in last layer or below src
        if up and (lrec == depth.size-1 or lrec > lsrc):
            Puu = np.full_like(Gam[:, :, lsrc, :], 0+0j)
            Pud = np.full_like(Gam[:, :, lsrc, :], 0+0j)
            continue
        # No downgoing field if rec is in first layer or above src
        if not up and (lrec == 0 or lrec < lsrc):
            Pdu = np.full_like(Gam[:, :, lsrc, :], 0+0j)
            Pdd = np.full_like(Gam[:, :, lsrc, :], 0+0j)
            continue

        # Swaps if up=True
        if up:
            dp, dm = dm, dp
            Rmp, Rpm = Rpm, Rmp
            first_layer, last_layer = last_layer, first_layer

        # Calculate Pu+, Pu-, Pd+, Pd-; rec in src layer; Eqs  81/82, A-8/A-9
        iGam = Gam[:, :, lsrc, :]
        if last_layer:  # If src/rec are in top (up) or bottom (down) layer
            Pd = Rmp*np.exp(-iGam*dm)
            Pu = np.full_like(Gam[:, :, lsrc, :], 0+0j)
        else:           # If src and rec are in any layer in between
            Ms = 1 - Rmp*Rpm*np.exp(-2*iGam*ds)
            Pd = Rmp/Ms*np.exp(-iGam*dm)
            Pu = Rmp/Ms*pm*Rpm*np.exp(-iGam*(ds+dp))

        # Store P's
        if up:
            Puu = Pu
            Pud = Pd
        else:
            Pdu = Pd
            Pdd = Pu

    # Return fields (up- and downgoing)
    return Puu, Pud, Pdu, Pdd
