"""

Installation & requirements
---------------------------

Just add the path to `empymod` to your python-path variable.

Alternatively, to install it in your python distribution (linux), run:

.. code:: bash

   python setup.py install

Required are python version 3 or higher and the modules `NumPy`, `SciPy`, and
`numexpr`.


Citation
--------

I am in the process of publishing an article regarding `empymod`, and I will
put the info here once it is reality. If you publish results for which you used
`empymod`, please consider citing this article. Also consider citing
[Hunziker_et_al_2015]_ and [Key_2012]_, without which `empymod` would not
exist.


License
-------

Copyright 2016-2017 Dieter Werthmüller

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

See the *LICENSE*-file in the root directory for a full reprint of the Apache
License.


Missing features
----------------

A list of things that should or could be added and improved, in decreasing
priority:

    - **Tests**, **tests**, and **more tests**: The modeller `empymod` is
      lacking an extensive testing suite. But it should have one.  This would
      ideally be combined with automated testing by, for instance, Travis. It
      should also include some proper benchmarks.

    - Kernel
        - Include `scipy.integrate.quad` as an additional Hankel transform.
          There are cases when both `QWE` and `FHT` struggle, e.g. at very
          short offsets with very high frequencies (GPR).
        - A `cython` or `numba` (pure C?) implementation of the `kernel` and
          the `transform` modules. Maybe not worth it, as it may improve speed,
          but decrease accessibility. Both at the same time would be nice. A
          fast C-version for calculations (inversions), and a Python-version to
          tinker with for interested folks.

    - More modelling routines:
        - convolution with a wavelet for GPR (proper version of `model.gpr`)
        - pure wavenumber output-routine (proper version of `model.wavenumber`)
        - various source-receiver arrangements (loops etc)
        - load and save functions to store and load model, together with all
          information.

    - Module to create Hankel filters (nice to have addition, mainly for
      educational purposes).

    - GUI frontend


Notice
------

This product includes software that was initially (till 01/2017) developed at
*The Mexican Institute of Petroleum IMP* (*Instituto Mexicano del Petróleo*,
http://www.imp.mx). The project was funded through *The Mexican National
Council of Science and Technology* (*Consejo Nacional de Ciencia y Tecnología*,
http://www.conacyt.mx).


This product is a derivative work of [Hunziker_et_al_2015]_ and [Key_2012]_,
and their publicly available software:


1. Hunziker, J., J. Thorbecke, and E. Slob, 2015, The electromagnetic response
   in a layered vertical transverse isotropic medium: A new look at an old
   problem: Geophysics, 80, F1-F18; DOI: `10.1190/geo2013-0411.1
   <http://dx.doi.org/10.1190/geo2013-0411.1>`_; Software:
   `software.seg.org/2015/0001 <http://software.seg.org/2015/0001>`_.

2. Key, K., 2012, Is the fast Hankel transform faster than quadrature?:
   Geophysics, 77, F21-F30; DOI: `10.1190/GEO2011-0237.1
   <http://dx.doi.org/10.1190/GEO2011-0237.1>`_; Software:
   `software.seg.org/2012/0003 <http://software.seg.org/2012/0003>`_.

Both pieces of software are published under the *SEG disclaimer*. Parts of the
modeller `emmod` from Hunziker et al, 2015, is furthermore released under the
*Common Public License Version 1.0 (CPL)*. See the *NOTICE*-file in the root
directory for more information and a reprint of the SEG disclaimer and the CPL.


Note on speed, memory, and accuracy
-----------------------------------
There is the usual trade-off between speed, memory, and accuracy. Very
generally speaking we can say that the *FHT* is faster than *QWE*, but *QWE* is
much easier on memory usage. I doubt you will ever run into memory issues with
*QWE*, whereas for *FHT* you might for ten thousands of offsets or hundreds of
layers. Furthermore, *QWE* allows you to control the accuracy.

There are two optimisation possibilities included via the ``opt``-flag:
parallelisation (``opt='parallel'``) and spline interpolation
(``opt='spline'``).  They are switched off by default. The optimization
``opt='parallel'`` only affects speed and memory usage, whereas
``opt='spline'`` also affects precision!

I am sure `empymod` could be made much faster with cleverer coding style or
with the likes of `cython` or `numba`. Suggestions and contributions are
welcomed!


Depths, Rotation, and Bipole
''''''''''''''''''''''''''''
**Depths**: Calculation of many source and receiver positions is fastest if
they remain at the same depth, as they can be calculated in one kernel-call. If
depths do change, one has to loop over them.

**Rotation**: Sources and receivers aligned along the principal axes x, y, and
z can be calculated in one kernel call. For arbitrary aligned di- or bipoles,
3 kernel calls are required. If source and receiver are arbitrary aligned,
3x3 hence 9 kernel calls are required.

**Bipole**: Bipole increase the calculation time by the amount of integration
points used. For a source and a receiver bipole with each 5 integration points
you need 5x5 hence 25 kernel calls. You can calculate it in 1 kernel call if
you set both integration points to 1, and hence calculate the bipole as if they
were dipoles at their centre.

**Example**: For 1 source and 10 receivers, all at the same depth, 1 kernel
call is required.  If all receivers are at different depths, 10 kernel calls
are required. If you make source and receivers bipoles with 5 integration
points, 250 kernel calls are required.  If you rotate the source arbitrary
horizontally, 500 kernel calls are required. If you rotate the receivers too,
in the horizontal plane, 1'000 kernel calls are required. If you rotate the
receivers also vertically, 1'500 kernel calls are required. If you rotate the
source vertically too, 2'250 kernel calls are required. So your calculation
will take 2'250 times longer! No matter how fast the kernel is, this will take
a long time. Therefore carefully plan how precise you want to define your
source and receiver bipoles.

.. table:: Example as a table for comparison: 1 source, 10 receiver (one or
           many frequencies).

    +----------------+--------+-------+------+-------+-------+------+---------+
    |                |    source bipole      |        receiver bipole         |
    +================+========+=======+======+=======+=======+======+=========+
    |**kernel calls**| intpts | theta |  phi |intpts | theta |  phi | diff. z |
    +----------------+--------+-------+------+-------+-------+------+---------+
    |              1 |      1 |  0/90 | 0/90 |     1 |  0/90 | 0/90 |       1 |
    +----------------+--------+-------+------+-------+-------+------+---------+
    |             10 |      1 |  0/90 | 0/90 |     1 |  0/90 | 0/90 |      10 |
    +----------------+--------+-------+------+-------+-------+------+---------+
    |            250 |      5 |  0/90 | 0/90 |     5 |  0/90 | 0/90 |      10 |
    +----------------+--------+-------+------+-------+-------+------+---------+
    |            500 |      5 |  arb. | 0/90 |     5 |  0/90 | 0/90 |      10 |
    +----------------+--------+-------+------+-------+-------+------+---------+
    |           1000 |      5 |  arb. | 0/90 |     5 |  arb. | 0/90 |      10 |
    +----------------+--------+-------+------+-------+-------+------+---------+
    |           1500 |      5 |  arb. | 0/90 |     5 |  arb. | arb. |      10 |
    +----------------+--------+-------+------+-------+-------+------+---------+
    |           2250 |      5 |  arb. | arb. |     5 |  arb. | arb. |      10 |
    +----------------+--------+-------+------+-------+-------+------+---------+


Parallelisation
'''''''''''''''
If ``opt = 'parallel'``, a good dozen of the most time-consuming statements are
calculated by using the `numexpr` package
(https://github.com/pydata/numexpr/wiki/Numexpr-Users-Guide).  These statements
are all in the `kernel`-functions `greenfct`, `reflections`, and `fields`, and
all involve :math:`\Gamma` in one way or another, often calculating square
roots or exponentials. As :math:`\Gamma` has dimensions (#frequencies,
#offsets, #layers, #lambdas), it can become fairly big.

This parallelisation will make `empymod` faster if you calculate a lot of
offsets/frequencies at once, but slower for few offsets/frequencies. Best
practice is to check first which one is faster. (You can use the included
`jupyter notebook`-benchmark.)


Spline interpolation
''''''''''''''''''''
If ``opt = 'spline'``, the so-called *lagged convolution* or *splined* variant
of the *FHT* (depending on ``htarg``) or the *splined* version of the *QWE* are
applied. The spline option should be used with caution, as it is an
interpolation and therefore less precise than the non-spline version. However,
it significantly speeds up *QWE*, and massively speeds up *FHT*. (The
`numexpr`-version of the spline option is slower than the pure spline one, and
therefore it is only possible to have either ``'parallel'`` or ``'spline'``
on.)

Setting ``opt = 'spline'`` is generally faster. Good speed-up is achieved for
*QWE* by setting ``maxint`` as low as possible. Also, the higher ``nquad`` is,
the higher the speed-up will be.  The variable ``pts_per_dec`` has also some
influence. For *FHT*, big improvements are achieved for long FHT-filters and
for many offsets/frequencies (thousands).  Additionally, spline minimizes
memory requirements a lot.  Speed-up is greater if all source-receiver angles
are identical.

`FHT`: Default for ``pts_per_dec = None``, which is the original *lagged
convolution*, where the spacing is defined by the filter-base, the transform is
carried out first followed by spline-interpolation. You can set this parameter
to an integer, which defines the number of points to evaluate per decade. In
this case the spline-interpolation is carried out first, followed by the
transformation. The original *lagged convolution* is generally the fastest for
a very good precision. However, by setting ``pts_per_dec`` appropriately one
can achieve higher precision, normally at the cost of speed.

.. warning::

    Keep in mind that it uses interpolation, and is therefore not as
    accurate as the non-spline version.  Use with caution and always compare
    with the non-spline version if you can apply the spline-version to your
    problem at hand!

Be aware that the `QWE`- and the `FHT`-Versions for the frequency-to-time
transformation *always* use the splined version and *always* loop over
offsets.

Looping
'''''''
By default, you can calculate many offsets and many frequencies all in one go,
vectorized (for the *FHT*), which is the default. The ``loop`` parameter gives
you the possibility to force looping over frequencies or offsets. This
parameter can have severe effects on both runtime and memory usage. Play around
with this factor to find the fastest version for your problem at hand. It
ALWAYS loops over frequencies if ``ht = 'QWE'`` or if ``opt = 'spline'``.  All
vectorized is very fast if there are few offsets or few frequencies. If there
are many offsets and many frequencies, looping over the smaller of the two will
be faster. Choosing the right looping together with ``opt = 'parallel'`` can
have a huge influence.

Vertical components
'''''''''''''''''''
It is advised to use ``xdirect = True`` (the default) if source and receiver
are in the same layer to calculate

    - the vertical electric field due to a vertical electric source,
    - configurations that involve vertical magnetic components (source or
      receiver),
    - all configurations when source and receiver depth are exactly the same.

The Hankel transforms methods are having sometimes difficulties transforming
these functions.

FFTLog
------

FFTLog is the logarithmic analogue to the Fast Fourier Transform FFT originally
proposed by [Talman_1978]_. The code used by `empymod` was published in
Appendix B of [Hamilton_2000]_ and is publicly available at
`casa.colorado.edu/~ajsh/FFTLog <http://casa.colorado.edu/~ajsh/FFTLog>`_.
From the `FFTLog`-website:

*FFTLog is a set of fortran subroutines that compute the fast Fourier or Hankel
(= Fourier-Bessel) transform of a periodic sequence of logarithmically spaced
points.*

FFTlog can be used for the Hankel as well as for the Fourier Transform, but
currently `empymod` uses it only for the Fourier transform. It uses a
simplified version of the python implementation of FFTLog, `pyfftlog`
(`github.com/prisae/pyfftlog <https://github.com/prisae/pyfftlog>`_).



.. |_| unicode:: 0xA0
   :trim:

References |_|
--------------

.. [Anderson_1975] Anderson, W.L., 1975, Improved digital filters for
   evaluating Fourier and Hankel transform integrals:
   USGS Unnumbered Series;
   `<http://pubs.usgs.gov/unnumbered/70045426/report.pdf>`_.
.. [Anderson_1979] Anderson, W. L., 1979, Numerical integration of related
   Hankel transforms of orders 0 and 1 by adaptive digital filtering:
   Geophysics, 44, 1287--1305; DOI: |_| `10.1190/1.1441007
   <http://dx.doi.org/10.1190/1.1441007>`_.
.. [Anderson_1982] Anderson, W. L., 1982, Fast Hankel transforms using
   related and lagged convolutions: ACM Trans. on Math. Softw. (TOMS), 8,
   344--368; DOI: |_| `10.1145/356012.356014
   <http://dx.doi.org/10.1145/356012.356014>`_.
.. [Gosh_1971] Ghosh, D. P., 1971, The application of linear filter theory to
   the direct interpretation of geoelectrical resistivity sounding
   measurements: Geophysical Prospecting, 19, 192--217;
   DOI: |_| `10.1111/j.1365-2478.1971.tb00593.x
   <http://dx.doi.org/10.1111/j.1365-2478.1971.tb00593.x>`_.
.. [Hamilton_2000] Hamilton, A. J. S., 2000, Uncorrelated modes of the
   non-linear power spectrum: Monthly Notices of the Royal Astronomical
   Society, 312, pages 257-284; DOI: |_| `10.1046/j.1365-8711.2000.03071.x
   <http://dx.doi.org/10.1046/j.1365-8711.2000.03071.x>`_; Website of FFTLog:
   `casa.colorado.edu/~ajsh/FFTLog <http://casa.colorado.edu/~ajsh/FFTLog>`_.
.. [Hunziker_et_al_2015] Hunziker, J., J. Thorbecke, and E. Slob, 2015, The
   electromagnetic response in a layered vertical transverse isotropic medium:
   A new look at an old problem: Geophysics, 80, F1--F18;
   DOI: |_| `10.1190/geo2013-0411.1
   <http://dx.doi.org/10.1190/geo2013-0411.1>`_;
   Software: `software.seg.org/2015/0001 <http://software.seg.org/2015/0001>`_.
.. [Key_2009] Key, K., 2009, 1D inversion of multicomponent, multifrequency
   marine CSEM data: Methodology and synthetic studies for resolving thin
   resistive layers: Geophysics, 74, F9--F20; DOI: |_| `10.1190/1.3058434
   <http://dx.doi.org/10.1190/1.3058434>`_.
   Software: `marineemlab.ucsd.edu/Projects/Occam/1DCSEM
   <http://marineemlab.ucsd.edu/Projects/Occam/1DCSEM>`_.
.. [Key_2012] Key, K., 2012, Is the fast Hankel transform faster than
   quadrature?: Geophysics, 77, F21--F30; DOI: |_| `10.1190/GEO2011-0237.1
   <http://dx.doi.org/10.1190/GEO2011-0237.1>`_;
   Software: `software.seg.org/2012/0003 <http://software.seg.org/2012/0003>`_.
.. [Kong_2007] Kong, F. N., 2007, Hankel transform filters for dipole antenna
   radiation in a conductive medium: Geophysical Prospecting, 55, 83--89;
   DOI: |_| `10.1111/j.1365-2478.2006.00585.x
   <http://dx.doi.org/10.1111/j.1365-2478.2006.00585.x>`_.
.. [Shanks_1955] Shanks, D., 1955, Non-linear transformations of divergent and
   slowly convergent sequences: Journal of Mathematics and Physics, 34, 1--42;
   DOI: |_| `10.1002/sapm19553411
   <http://dx.doi.org/10.1002/sapm19553411>`_.
.. [Slob_et_al_2010] Slob, E., J. Hunziker, and W. A. Mulder, 2010, Green's
   tensors for the diffusive electric field in a VTI half-space: PIER, 107,
   1--20: DOI: |_| `10.2528/PIER10052807
   <http://dx.doi.org/10.2528/PIER10052807>`_.
.. [Talman_1978] Talman, J. D., 1978, Numerical Fourier and Bessel transforms
    in logarithmic variables: Journal of Computational Physics, 29, pages
    35-48; DOI: |_| `10.1016/0021-9991(78)90107-9
    <http://dx.doi.org/10.1016/0021-9991(78)90107-9>`_.
.. [Trefethen_2000] Trefethen, L. N., 2000, Spectral methods in MATLAB: Society
   for Industrial and Applied Mathematics (SIAM), volume 10 of Software,
   Environments, and Tools, chapter 12, page 129;
   DOI: |_| `10.1137/1.9780898719598.ch12
   <http://dx.doi.org/10.1137/1.9780898719598.ch12>`_.
.. [Weniger_1989] Weniger, E. J., 1989, Nonlinear sequence transformations for
   the acceleration of convergence and the summation of divergent series:
   Computer Physics Reports, 10, 189--371;
   arXiv: |_| `abs/math/0306302 <https://arxiv.org/abs/math/0306302>`_.
.. [Wynn_1956] Wynn, P., 1956, On a device for computing the
   :math:`e_m(S_n)` tranformation: Math. Comput., 10, 91--96;
   DOI: |_| `10.1090/S0025-5718-1956-0084056-6
   <http://dx.doi.org/10.1090/S0025-5718-1956-0084056-6>`_.

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

# Import main modelling routines to make them available as primary functions
from .model import bipole, dipole, frequency, time
__all__ = ['bipole', 'dipole', 'frequency', 'time']
