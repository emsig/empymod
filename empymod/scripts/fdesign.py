r"""
:mod:`fdesign` -- Digital Linear Filter (DLF) design
====================================================

The add-on fdesign can be used to design digital linear filters for the Hankel
or Fourier transform, or for any linear transform ([Ghos70]_). For this
included or provided theoretical transform pairs can be used. Alternatively,
one can use the EM modeller empymod to use the responses to an arbitrary 1D
model as numerical transform pair.

More information can be found in the following places:

- The article about fdesign is in the repo
  https://github.com/empymod/article-fdesign
- Example notebooks to design a filter can be found in the repo
  https://github.com/empymod/empymod-examples

This filter designing tool uses the direct matrix inversion method as described
in [Kong07]_ and is based on scripts by [Key12]_. The whole project of
``fdesign`` started with the Matlab scripts from Kerry Key, which he used to
design his filters for [Key09]_, [Key12]_. Fruitful discussions with Evert Slob
and Kerry Key improved the add-on substantially.

Note that the use of empymod to create numerical transform pairs is, as of now,
only implemented for the Hankel transform.


Implemented analytical transform pairs
--------------------------------------

The following tables list the transform pairs which are implemented by default.
Any other transform pair can be provided as input. A transform pair is defined
in the following way:

.. code-block:: python

   from empymod.scripts import fdesign

   def my_tp_pair(var):
       '''My transform pair.'''

       def lhs(l):
           return func(l, var)

       def rhs(r):
           return func(r, var)

       return fdesign.Ghosh(name, lhs, rhs)

Here, ``name`` must be one of ``j0``, ``j1``, ``sin``, or ``cos``, depending
what type of transform pair it is. Additional variables are provided with
``var``. The evaluation points of the ``lhs`` are denoted by ``l``, and the
evaluation points of the ``rhs`` are denoted as ``r``. As an example here the
implemented transform pair ``j0_1``

.. code-block:: python

   def j0_1(a=1):
       '''Hankel transform pair J0_1 ([Ande75]_).'''

       def lhs(l):
           return l*np.exp(-a*l**2)

       def rhs(r):
           return np.exp(-r**2/(4*a))/(2*a)

       return Ghosh('j0', lhs, rhs)


Implemented Hankel transforms
-----------------------------

- ``j0_1`` [Ande75]_

  .. math::

      \int^\infty_0 l \exp\left(-al^2\right) J_0(lr) dl =
      \frac{\exp\left(\frac{-r^2}{4a}\right)}{2a}

- ``j0_2`` [Ande75]_

  .. math::

      \int^\infty_0 \exp\left(-al\right) J_0(lr) dl =
      \frac{1}{\sqrt{a^2+r^2}}

- ``j0_3`` [GuSi97]_

  .. math::

      \int^\infty_0 l\exp\left(-al\right) J_0(lr) dl =
      \frac{a}{(a^2 + r^2)^{3/2}}

- ``j0_4`` [ChCo82]_

  .. math::

      \int^\infty_0 \frac{l}{\beta} \exp\left(-\beta z_v \right)
      J_0(lr) dl =
      \frac{\exp\left(-\gamma R\right)}{R}

- ``j0_5`` [ChCo82]_

  .. math::

    \int^\infty_0 l \exp\left(-\beta z_v \right)
    J_0(lr) dl =
    \frac{ z_v (\gamma R + 1)}{R^3}\exp\left(-\gamma R\right)

- ``j1_1`` [Ande75]_

  .. math::

    \int^\infty_0 l^2 \exp\left(-al^2\right) J_1(lr) dl =
    \frac{r}{4a^2} \exp\left(-\frac{r^2}{4a}\right)


- ``j1_2`` [Ande75]_

  .. math::

    \int^\infty_0 \exp\left(-al\right) J_1(lr) dl =
    \frac{\sqrt{a^2+r^2}-a}{r\sqrt{a^2 + r^2}}

- ``j1_3`` [Ande75]_

  .. math::

    \int^\infty_0 l \exp\left(-al\right) J_1(lr) dl =
    \frac{r}{(a^2 + r^2)^{3/2}}

- ``j1_4`` [ChCo82]_

  .. math::

    \int^\infty_0 \frac{l^2}{\beta} \exp\left(-\beta z_v \right)
    J_1(lr) dl =
    \frac{r(\gamma R+1)}{R^3}\exp\left(-\gamma R\right)

- ``j1_5`` [ChCo82]_

  .. math::

    \int^\infty_0 l^2 \exp\left(-\beta z_v \right)
    J_1(lr) dl =
    \frac{r z_v(\gamma^2R^2+3\gamma R+3)}{R^5}\exp\left(-\gamma R\right)

Where

.. math:: a >0, r>0
.. math:: z_v = |z_{rec} - z_{src}|
.. math:: R = \sqrt{r^2 + z_v^2}
.. math:: \gamma = \sqrt{2j\pi\mu_0f/\rho}
.. math:: \beta = \sqrt{l^2 + \gamma^2}


Implemented Fourier transforms
------------------------------

- ``sin_1`` [Ande75]_

  .. math::

    \int^\infty_0 l\exp\left(-a^2l^2\right) \sin(lr) dl =
    \frac{\sqrt{\pi}r}{4a^3} \exp\left(-\frac{r^2}{4a^2}\right)

- ``sin_2`` [Ande75]_

  .. math::

    \int^\infty_0 \exp\left(-al\right) \sin(lr) dl =
    \frac{r}{a^2 + r^2}

- ``sin_3`` [Ande75]_

  .. math::

    \int^\infty_0 \frac{l}{a^2+l^2} \sin(lr) dl =
    \frac{\pi}{2} \exp\left(-ar\right)

- ``cos_1`` [Ande75]_

  .. math::

    \int^\infty_0 \exp\left(-a^2l^2\right) \cos(lr) dl =
    \frac{\sqrt{\pi}}{2a} \exp\left(-\frac{r^2}{4a^2}\right)

- ``cos_2`` [Ande75]_

  .. math::

    \int^\infty_0 \exp\left(-al\right) \cos(lr) dl =
    \frac{a}{a^2 + r^2}

- ``cos_3`` [Ande75]_

  .. math::

    \int^\infty_0 \frac{1}{a^2+l^2} \cos(lr) dl =
    \frac{\pi}{2a} \exp\left(-ar\right)


"""
# Copyright 2017-2019 Dieter Werthmüller
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

import os
import numpy as np
from copy import deepcopy as dc
from scipy.constants import mu_0
from scipy.optimize import brute, fmin_powell

# Optional imports
try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = False
    plt_msg = "* WARNING :: `matplotlib` is not installed, no figures shown."

from ..filters import DigitalFilter
from ..model import dipole, dipole_k
from ..filters import key_201_2009 as j0j1filt
from ..filters import key_201_CosSin_2012 as sincosfilt
from ..utils import printstartfinish, timedelta, default_timer

__all__ = ['design', 'save_filter', 'load_filter', 'plot_result',
           'print_result', 'Ghosh', 'j0_1', 'j0_2', 'j0_3', 'j0_4', 'j0_5',
           'j1_1', 'j1_2', 'j1_3', 'j1_4', 'j1_5', 'sin_1', 'sin_2', 'sin_3',
           'cos_1', 'cos_2', 'cos_3', 'empy_hankel']


# 1. PRINCIPAL FILTER DESIGNING ROUTINES

def design(n, spacing, shift, fI, fC=False, r=None, r_def=(1, 1, 2), reim=None,
           cvar='amp', error=0.01, name=None, full_output=False, finish=False,
           save=True, path='filters', verb=2, plot=1):
    r"""Digital linear filter (DLF) design

    This routine can be used to design digital linear filters for the Hankel or
    Fourier transform, or for any linear transform ([Ghos70]_). For this
    included or provided theoretical transform pairs can be used.
    Alternatively, one can use the EM modeller empymod to use the responses to
    an arbitrary 1D model as numerical transform pair.

    This filter designing tool uses the direct matrix inversion method as
    described in [Kong07]_ and is based on scripts by [Key12]_. The tool is an
    add-on to the electromagnetic modeller empymod [Wert17]_. Fruitful
    discussions with Evert Slob and Kerry Key improved the add-on
    substantially.

    Example notebooks of its usage can be found in the repo
    `github.com/empymod/empymod-examples
    <https://github.com/empymod/empymod-examples>`_.

    Parameters
    ----------
    n : int
        Filter length.

    spacing: float or tuple (start, stop, num)
        Spacing between filter points. If tuple, it corresponds to the input
        for np.linspace with endpoint=True.

    shift: float or tuple (start, stop, num)
        Shift of base from zero. If tuple, it corresponds to the input for
        np.linspace with endpoint=True.

    fI, fC : transform pairs
        Theoretical or numerical transform pair(s) for the inversion (I) and
        for the check of goodness (fC). fC is optional. If not provided, fI is
        used for both fI and fC.

    r : array, optional
        Right-hand side evaluation points for the check of goodness (fC).
        Defaults to r = np.logspace(0, 5, 1000), which are a lot of evaluation
        points, and depending on the transform pair way too long r's.

    r_def : tuple (add_left, add_right, factor), optional
        Definition of the right-hand side evaluation points r of the inversion.
        r is derived from the base values, default is (1, 1, 2).

        - rmin = log10(1/max(base)) - add_left
        - rmax = log10(1/min(base)) + add_right
        - r = logspace(rmin, rmax, factor*n)

    reim : np.real or np.imag, optional
        Which part of complex transform pairs is used for the inversion.
        Defaults to np.real.

    cvar : string {'amp', 'r'}, optional
        If 'amp', the inversion minimizes the amplitude. If 'r', the inversion
        maximizes the right-hand side evaluation point r. Defaults is 'amp'.

    error : float, optional
        Up to which relative error the transformation is considered good in the
        evaluation of the goodness. Default is 0.01 (1 %).

    name : str, optional
        Name of the filter. Defaults to dlf_+str(n).

    full_output : bool, optional
        If True, returns best filter and output from scipy.optimize.brute; else
        only filter. Default is False.

    finish : None, True, or callable, optional
        If callable, it is passed through to scipy.optimize.brute: minimization
        function to find minimize best result from brute-force approach.
        Default is None. You can simply provide True in order to use
        scipy.optimize.fmin_powell(). Set this to None if you are only
        interested in the actually provided spacing/shift-values.

    save : bool, optional
        If True, best filter is saved to plain text files in ./filters/. Can be
        loaded with fdesign.load_filter(name). If full, the inversion output is
        stored too. You can add '.gz' to `name`, which will then save the full
        inversion output in a compressed file instead of plain text.

    path : string, optional
        Absolute or relative path where output will be saved if `save=True`.
        Default is 'filters'.

    verb : {0, 1, 2}, optional
        Level of verbosity, default is 2:
            - 0: Print nothing.
            - 1: Print warnings.
            - 2: Print additional time, progress, and result

    plot : {0, 1, 2, 3}, optional
        Level of plot-verbosity, default is 1:
            - 0: Plot nothing.
            - 1: Plot brute-force result
            - 2: Plot additional theoretical transform pairs, and best inv.
            - 3: Plot additional inversion result
                 (can result in lots of plots depending on spacing and shift)
                 If you are using a notebook, use %matplotlib notebook to have
                 all inversion results appear in the same plot.

    Returns
    -------
    filter : empymod.filter.DigitalFilter instance
        Best filter for the input parameters.
    full : tuple
        Output from scipy.optimize.brute with full_output=True. (Returned when
        ``full_output`` is True.)

    """

    # === 1.  LET'S START ============
    t0 = printstartfinish(verb)

    # Check plot with matplotlib (soft dependency)
    if plot > 0 and not plt:
        plot = 0
        if verb > 0:
            print(plt_msg)

    # Ensure fI, fC are lists
    def check_f(f):
        if hasattr(f, 'name'):  # put into list if single tp
            f = [f, ]
        else:  # ensure list (works for lists, tuples, arrays)
            f = list(f)
        return f

    if not fC:  # copy fI if fC not provided
        fC = dc(fI)
    fI = check_f(fI)
    if fI[0].name == 'j2':
        print("* ERROR   :: j2 (jointly j0 and j1) is only implemented for " +
              "fC, not for fI!")
        raise ValueError('j2')
    fC = check_f(fC)

    # Check default input values
    if finish and not callable(finish):
        finish = fmin_powell
    if name is None:
        name = 'dlf_'+str(n)
    if r is None:
        r = np.logspace(0, 5, 1000)
    if reim not in [np.real, np.imag]:
        reim = np.real

    # Get spacing and shift slices, cast r
    ispacing = _ls2ar(spacing, 'spacing')
    ishift = _ls2ar(shift, 'shift')
    r = np.atleast_1d(r)

    # Initialize log-dict to keep track in brute-force minimization-function.
    log = {'cnt1': -1,   # Counter
           'cnt2': -1,   # %-counter;  v Total number of iterations v
           'totnr': np.arange(*ispacing).size*np.arange(*ishift).size,
           'time': t0,   # Timer
           'warn-r': 0}  # Warning for short r

    # === 2.  THEORETICAL MODEL rhs ============

    # Calculate rhs
    for i, f in enumerate(fC):
        fC[i].rhs = f.rhs(r)

    # Plot
    if plot > 1:
        _call_qc_transform_pairs(n, ispacing, ishift, fI, fC, r, r_def, reim)

    # === 3. RUN BRUTE FORCE OVER THE GRID ============
    full = brute(_get_min_val, (ispacing, ishift), full_output=True,
                 args=(n, fI, fC, r, r_def, error, reim, cvar, verb, plot,
                       log), finish=finish)

    # Add cvar-information to full: 0 for 'amp', 1 for 'r'
    if cvar == 'r':
        full += (1, )
    else:
        full += (0, )

    # Finish output from brute/fmin; depending if finish or not
    if verb > 1:
        print('')
        if callable(finish):
            print('')

    # Get best filter (full[0] contains spacing/shift of the best result).
    dlf = _calculate_filter(n, full[0][0], full[0][1], fI, r_def, reim, name)

    # If verbose, print result
    if verb > 1:
        print_result(dlf, full)

    # === 4.  FINISHED ============
    printstartfinish(verb, t0)

    # If plot, show result
    if plot > 0:
        print('* QC: Overview of brute-force inversion:')
        plot_result(dlf, full, False)
        if plot > 1:
            print('* QC: Inversion result of best filter (minimum amplitude):')
            _get_min_val(full[0], n, fI, fC, r, r_def, error, reim, cvar, 0,
                         plot+1, log)

    # Save if desired
    if save:
        if full_output:
            save_filter(name, dlf, full, path=path)
        else:
            save_filter(name, dlf, path=path)

    # Output, depending on full_output
    if full_output:
        return dlf, full
    else:
        return dlf


def save_filter(name, filt, full=None, path='filters'):
    r"""Save DLF-filter and inversion output to plain text files."""

    # First we'll save the filter using its internal routine.
    # This will create the directory ./filters if it doesn't exist already.
    filt.tofile(path)

    # If full, we store the inversion output
    if full:

        # Get file name
        path = os.path.abspath(path)
        if len(name.split('.')) == 2:
            suffix = '.gz'
        else:
            suffix = ''
        fullfile = os.path.join(path, name.split('.')[0]+'_full.txt' + suffix)

        # Get number of spacing and shift values
        nspace, nshift = full[3].shape

        # Create header
        header = 'Full inversion output from empymod.fdesign.design\n'
        header += 'Line 11: Nr of spacing values\n'
        header += 'Line 12: Nr of shift values\n'
        header += 'Line 13: Best spacing value\n'
        header += 'Line 14: Best shift value\n'
        header += 'Line 15: Min amplitude or max offset\n'

        header += 'Lines 16-{}: Spacing matrix '.format(nspace+15)
        header += '({} x {})\n'.format(nspace, nshift)

        header += 'Lines {}-{}: Spacing matrix '.format(nspace+16, 2*nspace+15)
        header += '({} x {})\n'.format(nspace, nshift)

        header += 'Lines {}-{}: Spacing '.format(2*nspace+16, 3*nspace+15)
        header += 'matrix ({} x {})\n'.format(nspace, nshift)

        header += 'Line {}: Integer: 0: min amp, 1: max r'.format(3*nspace+16)

        # Create arrays; put single values in arrays of nshift values
        nr_spacing = np.r_[nspace, np.zeros(nshift-1)]
        nr_shift = np.r_[nshift, np.zeros(nshift-1)]
        best_spacing = np.r_[full[0][0], np.zeros(nshift-1)]
        best_shift = np.r_[full[0][1], np.zeros(nshift-1)]
        min_value = np.r_[np.atleast_1d(full[1]), np.zeros(nshift-1)]
        min_max = np.r_[full[4], np.zeros(nshift-1)]

        # Collect all in one array
        fullsave = np.vstack((nr_spacing, nr_shift, best_spacing, best_shift,
                              min_value, full[2][0], full[2][1], full[3],
                              min_max))

        # Save array
        np.savetxt(fullfile, fullsave, header=header)


def load_filter(name, full=False, path='filters'):
    r"""Load saved DLF-filter and inversion output from text files."""

    # First we'll get the filter using its internal routine.
    filt = DigitalFilter(name.split('.')[0])
    filt.fromfile(path)

    # If full, we get the inversion output
    if full:
        # Try to get the inversion result. If files are not found, most likely
        # because they were not stored, we only return the filter
        try:
            # Get file name
            path = os.path.abspath(path)
            if len(name.split('.')) == 2:
                suffix = '.gz'
            else:
                suffix = ''
            fullfile = os.path.join(path, name.split('.')[0] +
                                    '_full.txt' + suffix)

            # Read data
            out = np.loadtxt(fullfile)

        except IOError:
            return filt

        # Collect inversion-result tuple
        nspace = int(out[0][0])
        nshift = int(out[1][0])

        space_shift_matrix = np.zeros((2, nspace, nshift))
        space_shift_matrix[0, :, :] = out[5:nspace+5, :]
        space_shift_matrix[1, :, :] = out[nspace+5:2*nspace+5, :]

        out = (np.array([out[2][0], out[3][0]]), out[4][0], space_shift_matrix,
               out[2*nspace+5:3*nspace+5, :], int(out[3*nspace+5, 0]))

        return filt, out
    else:
        return filt


# 2 PLOTTING ROUTINES (for QC or direct use)

# # 2.a Public plotting routines for QC or direct use

def plot_result(filt, full, prntres=True):
    r"""QC the inversion result.

    Parameters
    ----------
    - filt, full as returned from fdesign.design with full_output=True
    - If prntres is True, it calls fdesign.print_result as well.

    r"""
    # Check matplotlib (soft dependency)
    if not plt:
        print(plt_msg)
        return

    if prntres:
        print_result(filt, full)

    # Get spacing and shift values from full output of brute
    spacing = full[2][0, :, 0]
    shift = full[2][1, 0, :]

    # Get minimum field values from full output of brute
    minfield = np.squeeze(full[3])

    plt.figure("Brute force result", figsize=(9.5, 4.5))
    plt.subplots_adjust(wspace=.4, bottom=0.2)

    # Figure 1: Only if more than 1 spacing or more than 1 shift
    # Figure of minfield, depending if spacing/shift are vectors or floats
    if spacing.size > 1 or shift.size > 1:
        plt.subplot(121)
        if full[4] == 0:  # Min amp
            plt.title("Minimal recovered fields")
            ylabel = 'Minimal recovered amplitude (log10)'
            field = np.log10(minfield)
            cmap = plt.cm.viridis
        else:  # Max r
            plt.title("Maximum recovered r")
            ylabel = 'Maximum recovered r'
            field = 1/minfield
            cmap = plt.cm.viridis_r

        if shift.size == 1:    # (a) if only one shift value,
            plt.plot(spacing, field)
            plt.xlabel('Spacing')
            plt.ylabel(ylabel)

        elif spacing.size == 1:  # (b) if only one spacing value
            plt.plot(shift, field)
            plt.xlabel('Shift')
            plt.ylabel(ylabel)

        else:   # (c) if several spacing and several shift values
            field = np.ma.masked_where(np.isinf(minfield), field)
            plt.pcolormesh(shift, spacing, field, cmap=cmap)
            plt.ylabel('Spacing')
            plt.xlabel('Shift')
            plt.colorbar()

    # Figure 2: Filter values
    if spacing.size > 1 or shift.size > 1:
        plt.subplot(122)
    plt.title('Filter values of best filter')
    for attr in ['j0', 'j1', 'sin', 'cos']:
        if hasattr(filt, attr):
            plt.plot(np.log10(filt.base),
                     np.log10(np.abs(getattr(filt, attr))), '.-', lw=.5,
                     label='abs('+attr+')')
            plt.plot(np.log10(filt.base), np.log10(-getattr(filt, attr)), '.',
                     color='k', ms=4)
    plt.plot(np.inf, 0, '.', color='k', ms=4, label='Neg. values')
    plt.xlabel('Base (log10)')
    plt.ylabel('Abs(Amplitude) (log10)')
    plt.legend(loc='best')
    plt.gcf().canvas.draw()  # To force draw in notebook while running
    plt.show()


def print_result(filt, full=None):
    r"""Print best filter information.

    Parameters
    ----------
    - filt, full as returned from fdesign.design with full_output=True

    """
    print('   Filter length   : %d' % filt.base.size)
    print('   Best filter')

    if full:  # If full provided, we have more information
        if full[4] == 0:  # Min amp
            print('   > Min field     : %g' % full[1])
        else:  # Max amp
            r = 1/full[1]
            print('   > Max r         : %g' % r)
        spacing = full[0][0]
        shift = full[0][1]
    else:  # Print what we can without full
        n = filt.base.size
        a = filt.base[-1]
        b = filt.base[-2]
        spacing = np.log(a)-np.log(b)
        shift = np.log(a)-spacing*(n//2)
    print('   > Spacing       : %1.10g' % spacing)
    print('   > Shift         : %1.10g' % shift)
    print('   > Base min/max  : %e / %e' % (filt.base.min(), filt.base.max()))


# # 2.b Private plotting routines for QC

def _call_qc_transform_pairs(n, ispacing, ishift, fI, fC, r, r_def, reim):
    r"""QC the input transform pairs."""
    print('* QC: Input transform-pairs:')
    print('  fC: x-range defined through ``n``, ``spacing``, ``shift``, and ' +
          '``r``-parameters; b-range defined through ``r``-parameter.')
    print('  fI: x- and b-range defined through ``n``, ``spacing``' +
          ', ``shift``, and ``r_def``-parameters.')

    # Calculate min/max k, from minimum and maximum spacing/shift
    minspace = np.arange(*ispacing).min()
    maxspace = np.arange(*ispacing).max()
    minshift = np.arange(*ishift).min()
    maxshift = np.arange(*ishift).max()

    maxbase = np.exp(maxspace*(n//2) + maxshift)
    minbase = np.exp(maxspace*(-n//2+1) + minshift)

    # For fC-r  (k defined with same amount of points as r)
    kmax = maxbase/r.min()
    kmin = minbase/r.max()
    k = np.logspace(np.log10(kmin), np.log10(kmax) + minspace, r.size)

    # For fI-r
    rI = np.logspace(np.log10(1/maxbase) - r_def[0],
                     np.log10(1/minbase) + r_def[1], r_def[2]*n)
    kmaxI = maxbase/rI.min()
    kminI = minbase/rI.max()
    kI = np.logspace(np.log10(kminI), np.log10(kmaxI) + minspace,
                     r_def[2]*n)

    # Plot QC
    fig, axs = plt.subplots(figsize=(9.5, 6), nrows=2, ncols=2,
                            num="Transform pairs")
    axs = axs.ravel()
    plt.subplots_adjust(wspace=.3, hspace=.4)

    _plot_transform_pairs(fC, r, k, axs[:2], 'fC')
    if reim == np.real:
        tit = 'RE(fI)'
    else:
        tit = 'IM(fI)'
    _plot_transform_pairs(fI, rI, kI, axs[2:], tit)

    fig.canvas.draw()  # To force draw in notebook while running
    plt.show()


def _plot_transform_pairs(fCI, r, k, axes, tit):
    r"""Plot the input transform pairs."""

    # Plot lhs
    plt.sca(axes[0])
    plt.title('|' + tit + ' lhs|')
    for f in fCI:
        if f.name == 'j2':
            lhs = f.lhs(k)
            plt.loglog(k, np.abs(lhs[0]), lw=2, label='j0')
            plt.loglog(k, np.abs(lhs[1]), lw=2, label='j1')
        else:
            plt.loglog(k, np.abs(f.lhs(k)), lw=2, label=f.name)
    if tit != 'fC':
        plt.xlabel('l')
    plt.legend(loc='best')

    # Plot rhs
    plt.sca(axes[1])
    plt.title('|' + tit + ' rhs|')

    # Transform pair rhs
    for f in fCI:
        if tit == 'fC':
            plt.loglog(r, np.abs(f.rhs), lw=2, label=f.name)
        else:
            plt.loglog(r, np.abs(f.rhs(r)), lw=2, label=f.name)

    # Transform with Key
    for f in fCI:
        if f.name[1] in ['0', '1', '2']:
            filt = j0j1filt()
        else:
            filt = sincosfilt()
        kk = filt.base/r[:, None]
        if f.name == 'j2':
            lhs = f.lhs(kk)
            kr0 = np.dot(lhs[0], getattr(filt, 'j0'))/r
            kr1 = np.dot(lhs[1], getattr(filt, 'j1'))/r**2
            kr = kr0+kr1
        else:
            kr = np.dot(f.lhs(kk), getattr(filt, f.name))/r

        plt.loglog(r, np.abs(kr), '-.', lw=2, label=filt.name)

    if tit != 'fC':
        plt.xlabel('r')

    plt.legend(loc='best')


def _plot_inversion(f, rhs, r, k, imin, spacing, shift, cvar):
    r"""QC the resulting filter."""

    # Check matplotlib (soft dependency)
    if not plt:
        print(plt_msg)
        return

    plt.figure("Inversion result "+f.name, figsize=(9.5, 4))
    plt.subplots_adjust(wspace=.3, bottom=0.2)
    plt.clf()

    tk = np.logspace(np.log10(k.min()), np.log10(k.max()), r.size)

    plt.suptitle(f.name+'; Spacing ::'+str(spacing)+'; Shift ::'+str(shift))

    # Plot lhs
    plt.subplot(121)
    plt.title('|lhs|')
    if f.name == 'j2':
        lhs = f.lhs(tk)
        plt.loglog(tk, np.abs(lhs[0]), lw=2, label='Theoretical J0')
        plt.loglog(tk, np.abs(lhs[1]), lw=2, label='Theoretical J1')
    else:
        plt.loglog(tk, np.abs(f.lhs(tk)), lw=2, label='Theoretical')
    plt.xlabel('l')
    plt.legend(loc='best')

    # Plot rhs
    plt.subplot(122)
    plt.title('|rhs|')

    # Transform pair rhs
    plt.loglog(r, np.abs(f.rhs), lw=2, label='Theoretical')

    # Transform with filter
    plt.loglog(r, np.abs(rhs), '-.', lw=2, label='This filter')

    # Plot minimum amplitude or max r, respectively
    if cvar == 'amp':
        label = 'Min. Amp'
    else:
        label = 'Max. r'
    plt.loglog(r[imin], np.abs(rhs[imin]), 'go', label=label)

    plt.xlabel('r')
    plt.legend(loc='best')
    plt.gcf().canvas.draw()  # To force draw in notebook while running
    plt.show()


# 3. ANALYTICAL TRANSFORM PAIRS

class Ghosh:
    r"""Simple Class for Theoretical Transform Pairs.

    Named after D. P. Ghosh, honouring his 1970 Ph.D. thesis with which he
    introduced the digital filter method to geophysics ([Ghos70]_).
    """
    def __init__(self, name, lhs, rhs):
        r"""Add the filter name, lhs, and rhs."""
        self.name = name
        self.lhs = lhs
        self.rhs = rhs


# # 3.a Hankel J0 transform pairs

def j0_1(a=1):
    r"""Hankel transform pair J0_1 ([Ande75]_)."""

    def lhs(x):
        return x*np.exp(-a*x**2)

    def rhs(b):
        return np.exp(-b**2/(4*a))/(2*a)

    return Ghosh('j0', lhs, rhs)


def j0_2(a=1):
    r"""Hankel transform pair J0_2 ([Ande75]_)."""

    def lhs(x):
        return np.exp(-a*x)

    def rhs(b):
        return 1/np.sqrt(b**2 + a**2)

    return Ghosh('j0', lhs, rhs)


def j0_3(a=1):
    r"""Hankel transform pair J0_3 ([GuSi97]_)."""

    def lhs(x):
        return x*np.exp(-a*x)

    def rhs(b):
        return a/(b**2 + a**2)**1.5

    return Ghosh('j0', lhs, rhs)


def j0_4(f=1, rho=0.3, z=50):
    r"""Hankel transform pair J0_4 ([ChCo82]_).

    Parameters
    ----------
    f : float
        Frequency (Hz)
    rho : float
        Resistivity (Ohm.m)
    z : float
        Vertical distance between source and receiver (m)

    """

    gam = np.sqrt(2j*np.pi*mu_0*f/rho)

    def lhs(x):
        beta = np.sqrt(x**2 + gam**2)
        return x*np.exp(-beta*np.abs(z))/beta

    def rhs(b):
        R = np.sqrt(b**2 + z**2)
        return np.exp(-gam*R)/R

    return Ghosh('j0', lhs, rhs)


def j0_5(f=1, rho=0.3, z=50):
    r"""Hankel transform pair J0_5 ([ChCo82]_).

    Parameters
    ----------
    f : float
        Frequency (Hz)
    rho : float
        Resistivity (Ohm.m)
    z : float
        Vertical distance between source and receiver (m)

    """

    gam = np.sqrt(2j*np.pi*mu_0*f/rho)

    def lhs(x):
        beta = np.sqrt(x**2 + gam**2)
        return x*np.exp(-beta*np.abs(z))

    def rhs(b):
        R = np.sqrt(b**2 + z**2)
        return np.abs(z)*(gam*R + 1)*np.exp(-gam*R)/R**3

    return Ghosh('j0', lhs, rhs)


# # 3.b Hankel J1 transform pairs

def j1_1(a=1):
    r"""Hankel transform pair J1_1 ([Ande75]_)."""

    def lhs(x):
        return x**2*np.exp(-a*x**2)

    def rhs(b):
        return b/(4*a**2)*np.exp(-b**2/(4*a))

    return Ghosh('j1', lhs, rhs)


def j1_2(a=1):
    r"""Hankel transform pair J1_2 ([Ande75]_)."""

    def lhs(x):
        return np.exp(-a*x)

    def rhs(b):
        return (np.sqrt(b**2 + a**2) - a)/(b*np.sqrt(b**2 + a**2))

    return Ghosh('j1', lhs, rhs)


def j1_3(a=1):
    r"""Hankel transform pair J1_3 ([Ande75]_)."""

    def lhs(x):
        return x*np.exp(-a*x)

    def rhs(b):
        return b/(b**2 + a**2)**1.5

    return Ghosh('j1', lhs, rhs)


def j1_4(f=1, rho=0.3, z=50):
    r"""Hankel transform pair J1_4 ([ChCo82]_).

    Parameters
    ----------
    f : float
        Frequency (Hz)
    rho : float
        Resistivity (Ohm.m)
    z : float
        Vertical distance between source and receiver (m)

    """

    gam = np.sqrt(2j*np.pi*mu_0*f/rho)

    def lhs(x):
        beta = np.sqrt(x**2 + gam**2)
        return x**2*np.exp(-beta*np.abs(z))/beta

    def rhs(b):
        R = np.sqrt(b**2 + z**2)
        return b*(gam*R + 1)*np.exp(-gam*R)/R**3

    return Ghosh('j1', lhs, rhs)


def j1_5(f=1, rho=0.3, z=50):
    r"""Hankel transform pair J1_5 ([ChCo82]_).

    Parameters
    ----------
    f : float
        Frequency (Hz)
    rho : float
        Resistivity (Ohm.m)
    z : float
        Vertical distance between source and receiver (m)

    """

    gam = np.sqrt(2j*np.pi*mu_0*f/rho)

    def lhs(x):
        beta = np.sqrt(x**2 + gam**2)
        return x**2*np.exp(-beta*np.abs(z))

    def rhs(b):
        R = np.sqrt(b**2 + z**2)
        return np.abs(z)*b*(gam**2*R**2 + 3*gam*R + 3)*np.exp(-gam*R)/R**5

    return Ghosh('j1', lhs, rhs)


# # 3.c Fourier sine transform pairs

def sin_1(a=1):
    r"""Fourier sine transform pair sin_1 ([Ande75]_)."""

    def lhs(x):
        return x*np.exp(-a**2*x**2)

    def rhs(b):
        return np.sqrt(np.pi)*b*np.exp(-b**2/(4*a**2))/(4*a**3)

    return Ghosh('sin', lhs, rhs)


def sin_2(a=1):
    r"""Fourier sine transform pair sin_2 ([Ande75]_)."""

    def lhs(x):
        return np.exp(-a*x)

    def rhs(b):
        return b/(b**2 + a**2)

    return Ghosh('sin', lhs, rhs)


def sin_3(a=1):
    r"""Fourier sine transform pair sin_3 ([Ande75]_)."""

    def lhs(x):
        return x/(a**2 + x**2)

    def rhs(b):
        return np.pi*np.exp(-a*b)/2

    return Ghosh('sin', lhs, rhs)


# # 3.d Fourier cosine transform pairs

def cos_1(a=1):
    r"""Fourier cosine transform pair cos_1 ([Ande75]_)."""

    def lhs(x):
        return np.exp(-a**2*x**2)

    def rhs(b):
        return np.sqrt(np.pi)*np.exp(-b**2/(4*a**2))/(2*a)

    return Ghosh('cos', lhs, rhs)


def cos_2(a=1):
    r"""Fourier cosine transform pair cos_2 ([Ande75]_)."""

    def lhs(x):
        return np.exp(-a*x)

    def rhs(b):
        return a/(b**2 + a**2)

    return Ghosh('cos', lhs, rhs)


def cos_3(a=1):
    r"""Fourier cosine transform pair cos_3 ([Ande75]_)."""

    def lhs(x):
        return 1/(a**2 + x**2)

    def rhs(b):
        return np.pi*np.exp(-a*b)/(2*a)

    return Ghosh('cos', lhs, rhs)


# # 3.e Modeller

def empy_hankel(ftype, zsrc, zrec, res, freqtime, depth=None, aniso=None,
                epermH=None, epermV=None, mpermH=None, mpermV=None,
                htarg=None, verblhs=0, verbrhs=0):
    r"""Numerical transform pair with empymod.

    All parameters except ``ftype``, ``verblhs``, and ``verbrhs`` correspond to
    the input parameters to ``empymod.dipole``. See there for more information.

    Note that if depth=None or [], the analytical full-space solutions will be
    used (much faster).

    Parameters
    ----------
    ftype : str or list of strings
        Either of: {'j0', 'j1', 'j2', ['j0', 'j1']}

        - 'j0': Analyze J0-term with ab=11, angle=45°
        - 'j1': Analyze J1-term with ab=31, angle=0°
        - 'j2': Analyze J0- and J1-terms jointly with ab=12, angle=45°
        - ['j0', 'j1']: Same as calling empy_hankel twice, once with 'j0' and
                        one with 'j1'; can be provided like this to
                        fdesign.design.

    verblhs, verbrhs: int
        verb-values provided to empymod for lhs and rhs.

    Note that ftype='j2' only works for fC, not for fI.

    """

    # Loop over ftypes, if there are several
    if isinstance(ftype, list):
        out = []
        for f in ftype:
            out.append(empy_hankel(f, zsrc, zrec, res, freqtime, depth, aniso,
                                   epermH, epermV, mpermH, mpermV, htarg,
                                   verblhs, verbrhs))
        return out

    # Collect model
    model = {'src': [0, 0, zsrc],
             'depth': depth,
             'res': res,
             'aniso': aniso,
             'epermH': epermH,
             'epermV': epermV,
             'mpermH': mpermH,
             'mpermV': mpermV}

    # Finalize model depending on ftype
    if ftype == 'j0':  # J0: 11, 45°
        model['ab'] = 11
        x = 1/np.sqrt(2)
        y = 1/np.sqrt(2)

    elif ftype == 'j1':  # J1: 31, 0°
        model['ab'] = 31
        x = 1
        y = 0

    elif ftype == 'j2':  # J2: 12, 45°
        model['ab'] = 12
        x = 1/np.sqrt(2)
        y = 1/np.sqrt(2)

    # rhs: empymod.model.dipole
    # If depth=[], the analytical full-space solution will be used internally
    def rhs(r):
        out = dipole(rec=[r*x, r*y, zrec], ht='qwe', xdirect=True,
                     verb=verbrhs, htarg=htarg, freqtime=freqtime, **model)
        return out

    # lhs: empymod.model.dipole_k
    def lhs(k):
        lhs0, lhs1 = dipole_k(rec=[x, y, zrec], wavenumber=k, verb=verblhs,
                              freq=freqtime, **model)
        if ftype == 'j0':
            return lhs0
        elif ftype == 'j1':
            return lhs1
        elif ftype == 'j2':
            return (lhs0, lhs1)

    return Ghosh(ftype, lhs, rhs)


# 4. NON-USER-FACING ROUTINES

def _get_min_val(spaceshift, *params):
    r"""Calculate minimum resolved amplitude or maximum r."""

    # Get parameters from tuples
    spacing, shift = spaceshift
    n, fI, fC, r, r_def, error, reim, cvar, verb, plot, log = params

    # Get filter for these parameters
    dlf = _calculate_filter(n, spacing, shift, fI, r_def, reim, 'filt')

    # Calculate rhs-response with this filter
    k = dlf.base/r[:, None]

    # Loop over transforms
    for i, f in enumerate(fC):
        # Calculate lhs and rhs; rhs depends on ftype
        lhs = f.lhs(k)
        if f.name == 'j2':
            rhs0 = np.dot(lhs[0], getattr(dlf, 'j0'))/r
            rhs1 = np.dot(lhs[1], getattr(dlf, 'j1'))/r**2
            rhs = rhs0 + rhs1
        else:
            rhs = np.dot(lhs, getattr(dlf, f.name))/r

        # Get relative error
        rel_error = np.abs((rhs - f.rhs)/f.rhs)

        # Get indices where relative error is bigger than error
        imin0 = np.where(rel_error > error)[0]

        # Find first occurrence of failure
        if np.all(rhs == 0) or np.all(np.isnan(rhs)):
            # if all rhs are zeros or nans, the filter is useless
            imin0 = 0

        elif imin0.size == 0:
            # if imin0.size == 0:  # empty array, all rel_error < error.
            imin0 = rhs.size-1  # set to last r
            if verb > 0 and log['warn-r'] == 0:
                print('* WARNING :: all data have error < ' + str(error) +
                      '; choose larger r or set error-level higher.')
                log['warn-r'] = 1  # Only do this once

        else:
            # Kind of a dirty hack: Permit to jump up to four bad values,
            # resulting for instance from high rel_error from zero crossings
            # of the transform pair. Should be made an input argument or
            # generally improved.
            if imin0.size > 4:
                imin0 = np.max([0, imin0[4]-5])
            else:  # just take the first one (no jumping allowed; normal case)
                imin0 = np.max([0, imin0[0]-1])
            # Note that both version yield the same result if the failure is
            # consistent.

        # Depending on cvar, store minimum amplitude or 1/maxr
        if cvar == 'amp':
            min_val0 = np.abs(rhs[imin0])
        else:
            min_val0 = 1/r[imin0]

        # Check if this inversion is better than previous ones
        if i == 0:  # First run, store these values
            imin = dc(imin0)
            min_val = dc(min_val0)
        else:  # Replace imin, min_val if this one is better
            if min_val0 > min_val:
                min_val = dc(min_val0)
                imin = dc(imin0)

        # QC plot
        if plot > 2:
            _plot_inversion(f, rhs, r, k, imin0, spacing, shift, cvar)

    # If verbose, print progress
    if verb > 1:
        log = _print_count(log)

    # If there is no point with rel_error < error (imin=0) it returns np.inf.
    return np.where(imin == 0, np.inf, min_val)


def _calculate_filter(n, spacing, shift, fI, r_def, reim, name):
    r"""Calculate filter for this spacing, shift, n."""

    # Base :: For this n/spacing/shift
    base = np.exp(spacing*(np.arange(n)-n//2) + shift)

    # r :: Start/end is defined by base AND r_def[0]/r_def[1]
    #      Overdetermined system if r_def[2] > 1
    r = np.logspace(np.log10(1/np.max(base)) - r_def[0],
                    np.log10(1/np.min(base)) + r_def[1], r_def[2]*n)

    # k :: Get required k-values (matrix of shape (r.size, base.size))
    k = base/r[:, None]

    # Create filter instance
    dlf = DigitalFilter(name.split('.')[0])
    dlf.base = base
    dlf.factor = np.around(np.average(base[1:]/base[:-1]), 15)

    # Loop over transforms
    for f in fI:
        # Calculate lhs and rhs for inversion
        lhs = reim(f.lhs(k))
        rhs = reim(f.rhs(r)*r)

        # Calculate filter values: Solve lhs*J=rhs using linalg.qr.
        # If factoring fails (qr) or if matrix is singular or square (solve) it
        # will raise a LinAlgError. Error is ignored and zeros are returned
        # instead.
        try:
            qq, rr = np.linalg.qr(lhs)
            J = np.linalg.solve(rr, rhs.dot(qq))
        except np.linalg.LinAlgError:
            J = np.zeros((base.size,))

        setattr(dlf, f.name, J)

    return dlf


def _ls2ar(inp, strinp):
    r"""Convert float or linspace-input to arange/slice-input for brute."""

    # Check if input is 1 or 3 elements (float, list, tuple, array)
    if np.size(inp) == 1:
        start = np.squeeze(inp)
        stop = start+1
        num = 1
    elif np.size(inp) == 3:
        start = inp[0]
        stop = inp[1]
        num = inp[2]
    else:
        print("* ERROR   :: <"+strinp+"> must be a float or a tuple of 3 " +
              "elements (start, stop, num); <"+strinp+" provided: " + str(inp))
        raise ValueError(strinp)

    # Re-arrange it to be compatible with np.arange/slice for brute
    if num < 2 or start == stop:
        stop = start
        step = 1
    else:
        step = (stop-start)/(num-1)
    return (start, stop+step/2, step)


def _print_count(log):
    r"""Print run-count information."""
    log['cnt2'] += 1                   # Current number
    cp = log['cnt2']/log['totnr']*100  # Percentage

    if log['cnt2'] == 0:  # Not sure about this; brute seems to call the
        pass              # function with the first arguments twice...

    elif log['cnt2'] > log['totnr']:  # fmin-status
        print("   fmin  fct calls : %d" % (log['cnt2']-log['totnr']), end='\r')

    elif int(cp) > log['cnt1'] or cp < 1 or log['cnt2'] == log['totnr']:
        # Get seconds since start
        sec = int(default_timer() - log['time'])

        # Get estimate of remaining time, as string
        tleft = str(timedelta(seconds=int(100*sec/cp - sec)))

        # Print progress
        pstr = ("   brute fct calls : %d/%d"
                % (log['cnt2'], log['totnr']))
        if log['totnr'] > 100:
            pstr += (" (%d %%); est: %s        " % (cp, tleft))
        print(pstr, end='\r')

        if log['cnt2'] == log['totnr']:
            # Empty previous line
            print(" "*len(pstr), end='\r')

            # Print final brute-message
            print("   brute fct calls : %d" % log['totnr'])

        # Update percentage cnt1
        log['cnt1'] = cp

    return log
