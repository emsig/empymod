"""

Theory
------

The code is principally based on

- [Hunziker_et_al_2015]_ for the wavenumber-domain calculation (``kernel``),
- [Key_2012]_ for the DLF and QWE transforms,
- [Slob_et_al_2010]_ for the analytical half-space solutions, and
- [Hamilton_2000]_ for the FFTLog.

See these publications and all the others given in the references_, if you are
interested in the theory on which empymod is based. Another good reference is
[Ziolkowski_and_Slob]_, which will be published in late 2018. The book derives
in great detail the equations for layered-Earth CSEM modelling.


Installation
------------

You can install empymod either via ``conda``:

.. code-block:: console

   conda install -c prisae empymod

or via ``pip``:

.. code-block:: console

   pip install empymod

Required are Python version 3.4 or higher and the modules ``NumPy`` and
``SciPy``. The module ``numexpr`` is required additionally (built with Intel's
VML) if you want to run parts of the kernel in parallel.

The modeller empymod comes with add-ons (``empyscripts``). These add-ons
provide some very specific, additional functionalities. To install them just
follow the instructions for ``empymod``, replacing ``empymod`` with
``empyscripts`` in the command. You can find more information regarding the
add-ons directly on `github.com/empymod/empyscripts
<https://github.com/empymod/empyscripts>`_.

If you are new to Python I recommend using a Python distribution, which will
ensure that all dependencies are met, specifically properly compiled versions
of ``NumPy`` and ``SciPy``; I recommend using
`Anaconda <https://www.anaconda.com/download>`_.  If you install
[Anaconda](https://www.anaconda.com/download). If you install Anaconda you can
simply start the *Anaconda Navigator*, add the channel ``prisae`` and both
``empymod`` and ``empyscripts`` will appear in the package list and can be
installed with a click.

.. warning::

    Do not use ``scipy == 0.19.0``. It has a memory leak in ``quad``, see
    `github.com/scipy/scipy/pull/7216
    <https://github.com/scipy/scipy/pull/7216>`_. So if you use QUAD (or
    potentially QWE) in any of your transforms you might see your memory usage
    going through the roof.


The structure of empymod is:

- **model.py**: EM modelling routines.
- **utils.py**: Utilities for ``model`` such as checking input parameters.
- **kernel.py**: Kernel of ``empymod``, calculates the wavenumber-domain
  electromagnetic response. Plus analytical, frequency-domain full- and
  half-space solutions.
- **transform.py**: Methods to carry out the required Hankel transform from
  wavenumber to space domain and Fourier transform from frequency to time
  domain.
- **filters.py**: Filters for the *Digital Linear Filters* method DLF (Hankel
  and Fourier transforms).


Usage/Examples
--------------

A good starting point is [Werthmuller_2017b]_, and more information can be
found in [Werthmuller_2017]_. There are a lot of examples of its usage
available, in the form of Jupyter notebooks. Have a look at the following
repositories:

- Example notebooks: https://github.com/empymod/example-notebooks,
- Geophysical Tutoriol TLE: https://github.com/empymod/article-tle2017, and
- Numerical examples of [Ziolkowski_and_Slob]_:
  https://github.com/empymod/csem-ziolkowski-and-slob.

The main modelling routines is ``bipole``, which can calculate the
electromagnetic frequency- or time-domain field due to arbitrary finite
electric or magnetic bipole sources, measured by arbitrary finite electric or
magnetic bipole receivers. The model is defined by horizontal resistivity and
anisotropy, horizontal and vertical electric permittivities and horizontal and
vertical magnetic permeabilities. By default, the electromagnetic response is
normalized to source and receiver of 1 m length, and source strength of 1 A.

A simple frequency-domain example, with most of the parameters left at the
default value:

.. code-block:: python

    >>> import numpy as np
    >>> from empymod import bipole
    >>> # x-directed bipole source: x0, x1, y0, y1, z0, z1
    >>> src = [-50, 50, 0, 0, 100, 100]
    >>> # x-directed dipole source-array: x, y, z, azimuth, dip
    >>> rec = [np.arange(1, 11)*500, np.zeros(10), 200, 0, 0]
    >>> # layer boundaries
    >>> depth = [0, 300, 1000, 1050]
    >>> # layer resistivities
    >>> res = [1e20, .3, 1, 50, 1]
    >>> # Frequency
    >>> freq = 1
    >>> # Calculate electric field due to an electric source at 1 Hz.
    >>> # [msrc = mrec = True (default)]
    >>> EMfield = bipole(src, rec, depth, res, freq, verb=4)
    :: empymod START  ::
    ~
       depth       [m] :  0 300 1000 1050
       res     [Ohm.m] :  1E+20 0.3 1 50 1
       aniso       [-] :  1 1 1 1 1
       epermH      [-] :  1 1 1 1 1
       epermV      [-] :  1 1 1 1 1
       mpermH      [-] :  1 1 1 1 1
       mpermV      [-] :  1 1 1 1 1
       frequency  [Hz] :  1
       Hankel          :  DLF (Fast Hankel Transform)
         > Filter      :  Key 201 (2009)
         > DLF type    :  Standard
       Kernel Opt.     :  None
       Loop over       :  None (all vectorized)
       Source(s)       :  1 bipole(s)
         > intpts      :  1 (as dipole)
         > length  [m] :  100
         > x_c     [m] :  0
         > y_c     [m] :  0
         > z_c     [m] :  100
         > azimuth [°] :  0
         > dip     [°] :  0
       Receiver(s)     :  10 dipole(s)
         > x       [m] :  500 - 5000 : 10  [min-max; #]
                       :  500 1000 1500 2000 2500 3000 3500 4000 4500 5000
         > y       [m] :  0 - 0 : 10  [min-max; #]
                       :  0 0 0 0 0 0 0 0 0 0
         > z       [m] :  200
         > azimuth [°] :  0
         > dip     [°] :  0
       Required ab's   :  11
    ~
    :: empymod END; runtime = 0:00:00.005536 :: 1 kernel call(s)
    ~
    >>> print(EMfield)
    [  1.68809346e-10 -3.08303130e-10j  -8.77189179e-12 -3.76920235e-11j
      -3.46654704e-12 -4.87133683e-12j  -3.60159726e-13 -1.12434417e-12j
       1.87807271e-13 -6.21669759e-13j   1.97200208e-13 -4.38210489e-13j
       1.44134842e-13 -3.17505260e-13j   9.92770406e-14 -2.33950871e-13j
       6.75287598e-14 -1.74922886e-13j   4.62724887e-14 -1.32266600e-13j]

Contributing
------------

New contributions, bug reports, or any kind of feedback is always welcomed!
Have a look at the Roadmap-section to get an idea of things that could be
implemented. The best way for interaction is at https://github.com/empymod.
If you prefer to contact me outside of GitHub use the contact form on my
personal website, https://werthmuller.org.

To install empymod from source, you can download the latest version from GitHub
and either add the path to ``empymod`` to your python-path variable, or install
it in your python distribution via:

.. code-block:: console

   python setup.py install

Please make sure your code follows the pep8-guidelines by using, for instance,
the python module ``flake8``, and also that your code is covered with
appropriate tests. Just get in touch if you have any doubts.


The modeller comes with a test suite using ``pytest``. If you want to run the
tests, just install ``pytest`` and run it within the ``empymod``-top-directory.

.. code-block:: console

    > pip install pytest coveralls pytest-flake8
    > # and then
    > cd to/the/empymod/folder  # Ensure you are in the right directory,
    > ls -d */                  # your output should look the same.
    docs/  empymod/  tests/
    > # pytest will find the tests, which are located in the tests-folder.
    > # simply run
    > pytest --cov=empymod --flake8

It should run all tests successfully. Please let me know if not!

Note that installations of ``empymod`` via conda or pip do not have the
test-suite included. To run the test-suite you must download ``empymod`` from
GitHub.


Transforms
----------

Included **Hankel transforms**:

- Digital Linear Filters *DLF*
- Quadrature with Extrapolation *QWE*
- Adaptive quadrature *QUAD*

Included **Fourier transforms**:

- Digital Linear Filters *DLF*
- Quadrature with Extrapolation *QWE*
- Logarithmic Fast Fourier Transform *FFTLog*
- Fast Fourier Transform *FFT*


FFTLog
''''''

FFTLog is the logarithmic analogue to the Fast Fourier Transform FFT originally
proposed by [Talman_1978]_. The code used by ``empymod`` was published in
Appendix B of [Hamilton_2000]_ and is publicly available at
`casa.colorado.edu/~ajsh/FFTLog <http://casa.colorado.edu/~ajsh/FFTLog>`_.
From the ``FFTLog``-website:

*FFTLog is a set of fortran subroutines that compute the fast Fourier or Hankel
(= Fourier-Bessel) transform of a periodic sequence of logarithmically spaced
points.*

FFTlog can be used for the Hankel as well as for the Fourier Transform, but
currently ``empymod`` uses it only for the Fourier transform. It uses a
simplified version of the python implementation of FFTLog, ``pyfftlog``
(`github.com/prisae/pyfftlog <https://github.com/prisae/pyfftlog>`_).

[Haines_and_Jones_1988]_ proposed a logarithmic Fourier transform
(abbreviated by the authors as LFT) for electromagnetic geophysics, also based
on [Talman_1978]_. I do not know if Hamilton was aware of the work by Haines
and Jones. The two publications share as reference only the original paper by
Talman, and both cite a publication of Anderson; Hamilton cites
[Anderson_1982]_, and Haines and Jones cite [Anderson_1979]_. Hamilton probably
never heard of Haines and Jones, as he works in astronomy, and Haines and Jones
was published in the *Geophysical Journal*.

Logarithmic FFTs are not widely used in electromagnetics, as far as I know,
probably because of the ease, speed, and generally sufficient precision of the
digital filter methods with sine and cosine transforms ([Anderson_1975]_).
However, comparisons show that FFTLog can be faster and more precise than
digital filters, specifically for responses with source and receiver at the
interface between air and subsurface. Credit to use FFTLog in electromagnetics
goes to David Taylor who, in the mid-2000s, implemented FFTLog into the forward
modellers of the company Multi-Transient ElectroMagnetic (MTEM Ltd, later
Petroleum Geo-Services PGS). The implementation was driven by land responses,
where FFTLog can be much more precise than the filter method for very early
times.


Notes on Fourier Transform
''''''''''''''''''''''''''

The Fourier transform to obtain the space-time domain impulse response from the
complex-valued space-frequency response can be calculated by either a
cosine transform with the real values, or a sine transform with the imaginary
part,

.. math::

    E(r, t)^\\text{Impulse} &= \ \\frac{2}{\pi}\int^\infty_0 \Re[E(r, \omega)]\
                        \cos(\omega t)\ \\text{d}\omega \ , \\\\
            &= -\\frac{2}{\pi}\int^\infty_0 \Im[E(r, \omega)]\
                \sin(\omega t)\ \\text{d}\omega \ ,

see, e.g., [Anderson_1975]_ or [Key_2012]_. Quadrature-with-extrapolation,
FFTLog, and obviously the sine/cosine-transform all make use of this split.

To obtain the step-on response the frequency-domain result is first divided
by :math:`i\omega`, in the case of the step-off response it is additionally
multiplied by -1. The impulse-response is the time-derivative of the
step-response,

.. math::

    E(r, t)^\\text{Impulse} =
                        \\frac{\partial\ E(r, t)^\\text{step}}{\partial t}\ .

Using :math:`\\frac{\partial}{\partial t} \Leftrightarrow i\omega` and going
the other way, from impulse to step, leads to the divison by :math:`i\omega`.
(This only holds because we define in accordance with the causality principle
that :math:`E(r, t \le 0) = 0`).

With the sine/cosine transform (``ft='ffht'/'sin'/'cos'``) you can choose which
one you want for the impulse responses. For the switch-on response, however,
the sine-transform is enforced, and equally the cosine transform for the
switch-off response. This is because these two do not need to now the field at
time 0, :math:`E(r, t=0)`.

The Quadrature-with-extrapolation and FFTLog are hard-coded to use the cosine
transform for step-off responses, and the sine transform for impulse and
step-on responses. The FFT uses the full complex-valued response at the moment.

For completeness sake, the step-on response is given by

.. math::

    E(r, t)^\\text{Step-on} = - \\frac{2}{\pi}\int^\infty_0
                            \Im\\left[\\frac{E(r,\omega)}{i \omega}\\right]\
                            \sin(\omega t)\ \\text{d}\omega \ ,

and the step-off by

.. math::

    E(r, t)^\\text{Step-off} = - \\frac{2}{\pi}\int^\infty_0
                             \Re\\left[\\frac{E(r,\omega)}{i\omega}\\right]\
                             \cos(\omega t)\ \\text{d}\omega \ .


Note on speed, memory, and accuracy
-----------------------------------

There is the usual trade-off between speed, memory, and accuracy. Very
generally speaking we can say that the *DLF* is faster than *QWE*, but *QWE* is
much easier on memory usage. *QWE* allows you to control the accuracy. A
standard quadrature in the form of *QUAD* is also provided. *QUAD* is generally
orders of magnitudes slower, and more fragile depending on the input arguments.
However, it can provide accurate results where *DLF* and *QWE* fail.

The kernel can run in parallel using `numexpr`. This option is activated by
setting ``opt='parallel'``. It is switched off by default.

I am sure ``empymod`` could be made much faster with cleverer coding style or
with the likes of ``cython`` or ``numba``. Suggestions and contributions are
welcomed!


Memory
''''''
By default ``empymod`` will try to carry out the calculation in one go, without
looping. If your model has many offsets and many frequencies this can be heavy
on memory usage. Even more so if you are calculating time-domain responses for
many times. If you are running out of memory, you should use either
``loop='off'`` or ``loop='freq'`` to loop over offsets or frequencies,
respectively. Use ``verb=3`` to see how many offsets and how many frequencies
are calculated internally.



Depths, Rotation, and Bipole
''''''''''''''''''''''''''''
**Depths**: Calculation of many source and receiver positions is fastest if
they remain at the same depth, as they can be calculated in one kernel-call. If
depths do change, one has to loop over them. Note: Sources or receivers placed
on a layer interface are considered in the upper layer.

**Rotation**: Sources and receivers aligned along the principal axes x, y, and
z can be calculated in one kernel call. For arbitrary oriented di- or bipoles,
3 kernel calls are required. If source and receiver are arbitrary oriented,
9 (3x3) kernel calls are required.

**Bipole**: Bipoles increase the calculation time by the amount of integration
points used. For a source and a receiver bipole with each 5 integration points
you need 25 (5x5) kernel calls. You can calculate it in 1 kernel call if you
set both integration points to 1, and therefore calculate the bipole as if they
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
    |**kernel calls**| intpts |azimuth|  dip |intpts |azimuth|  dip | diff. z |
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
calculated by using the ``numexpr`` package
(https://github.com/pydata/numexpr/wiki/Numexpr-Users-Guide).  These statements
are all in the ``kernel``-functions ``greenfct``, ``reflections``, and
``fields``, and all involve :math:`\Gamma` in one way or another, often
calculating square roots or exponentials. As :math:`\Gamma` has dimensions
(#frequencies, #offsets, #layers, #lambdas), it can become fairly big.

The package ``numexpr`` has to be built with Intel's VML, otherwise it won't be
used. You can check if it uses VML with

.. code-block:: python

    >>> import numexpr
    >>> numexpr.use_vml

The module ``numexpr`` uses by default all available cores up to a maximum of
8. You can change this behaviour to a lower or a higher value with the
following command (in the example it is changed to 4):

.. code-block:: python

    >>> import numexpr
    >>> numexpr.set_num_threads(4)

This parallelisation will make ``empymod`` faster if you calculate a lot of
offsets/frequencies at once, but slower for few offsets/frequencies. Best
practice is to check first which one is faster. (You can use the
benchmark-notebook in the `empymod/example-notebooks
<https://github.com/empymod/example-notebooks>`_-repository.)

Spline interpolation
''''''''''''''''''''
Both Hankel and Fourier DLF have three options, which can be controlled via
the ``htarg['pts_per_dec']`` and ``ftarg['pts_per_dec']`` parameters:
    - ``pts_per_dec=0`` : *Standard DLF*;
    - ``pts_per_dec<0`` : *Lagged Convolution DLF*: Spacing defined by filter
      base, interpolation is carried out in the input domain;
    - ``pts_per_dec>0`` : *Splined DLF*: Spacing defined by ``pts_per_dec``,
      interpolation is carried out in the output domain.

Similarly, interpolation can be used for ``QWE`` by setting ``pts_per_dec`` to
a value bigger than 0.

The spline option should be used with caution, as it is an interpolation and
therefore less precise than the non-spline version. However, it significantly
speeds up *QWE*, and massively speeds up *DLF*. (Note that the
``numexpr``-version of the spline option is slower than the pure spline one.)

Using the splined options is generally faster. Good speed-up is achieved for
*QWE* by setting ``maxint`` as low as possible. Also, the higher ``nquad`` is,
the higher the speed-up will be.  The variable ``pts_per_dec`` has also some
influence. For *DLF*, big improvements are achieved for long DLF-filters and
for many offsets/frequencies (thousands).  Additionally, spline minimizes
memory requirements a lot.  Speed-up is greater if all source-receiver angles
are identical.

*DLF*: Default for Hankel DLF ``pts_per_dec = 0``, which is the original
*lagged convolution*, where the spacing is defined by the filter-base, the
transform is carried out first followed by spline-interpolation. You can set
this parameter to an integer, which defines the number of points to evaluate
per decade. In this case the spline-interpolation is carried out first,
followed by the transformation. The original *lagged convolution* is generally
the fastest for a very good precision. However, by setting ``pts_per_dec``
appropriately one can achieve higher precision, normally at the cost of speed.

.. warning::

    Keep in mind that it uses interpolation, and is therefore not as
    accurate as the non-spline version.  Use with caution and always compare
    with the non-spline version if you can apply the spline-version to your
    problem at hand!

Be aware that *QUAD* (Hankel transform) *always* use the splined version and
*always* loop over offsets. The Fourier transforms *FFTlog*, *QWE*, and *FFT*
always use interpolation too, either in the frequency or in the time domain.
With the *DLF* Fourier transform (sine and cosine transforms) you can choose
between no interpolation and interpolation (splined or lagged).

The splined versions of *QWE* check whether the ratio of any two adjacent
intervals is above a certain threshold (steep end of the wavenumber or
frequency spectrum). If it is, it carries out *QUAD* for this interval instead
of *QWE*. The threshold is stored in ``diff_quad``, which can be changed within
the parameter ``htarg`` and ``ftarg``.

For a graphical explanation of the differences between standard DLF, lagged
convolution DLF, and splined DLF for the Hankel and the Fourier transforms
see the notebook ``7a_DLF-Standard-Lagged-Splined`` in the
`example-notebooks <https://github.com/empymod/example-notebooks>`_ repository.

Looping
'''''''
By default, you can calculate many offsets and many frequencies
all in one go, vectorized (for the *DLF*), which is the default. The ``loop``
parameter gives you the possibility to force looping over frequencies or
offsets. This parameter can have severe effects on both runtime and memory
usage. Play around with this factor to find the fastest version for your
problem at hand. It ALWAYS loops over frequencies if ``ht = 'QWE'/'QUAD'`` or
if ``ht = 'FHT'`` and ``pts_per_dec<0`` (Lagged Convolution Hankel DLF). All
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


License
-------

Copyright 2016-2018 Dieter Werthmüller

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

See the ``LICENSE``- and ``NOTICE``-files on GitHub for more information.

.. note::

    This software was initially (till 01/2017) developed with funding from *The
    Mexican National Council of Science and Technology* (*Consejo Nacional de
    Ciencia y Tecnología*, http://www.conacyt.gob.mx), carried out at *The
    Mexican Institute of Petroleum IMP* (*Instituto Mexicano del Petróleo*,
    http://www.gob.mx/imp).


.. |_| unicode:: 0xA0
   :trim:

.. _references:

References |_|
--------------

.. [Anderson_1975] Anderson, W.L., 1975, Improved digital filters for
   evaluating Fourier and Hankel transform integrals:
   USGS Unnumbered Series;
   `<http://pubs.usgs.gov/unnumbered/70045426/report.pdf>`_.
.. [Anderson_1979] Anderson, W. L., 1979, Numerical integration of related
   Hankel transforms of orders 0 and 1 by adaptive digital filtering:
   Geophysics, 44, 1287--1305; DOI: |_| `10.1190/1.1441007
   <http://doi.org/10.1190/1.1441007>`_.
.. [Anderson_1982] Anderson, W. L., 1982, Fast Hankel transforms using
   related and lagged convolutions: ACM Trans. on Math. Softw. (TOMS), 8,
   344--368; DOI: |_| `10.1145/356012.356014
   <http://doi.org/10.1145/356012.356014>`_.
.. [Ghosh_1971] Ghosh, D. P., 1971, The application of linear filter theory to
   the direct interpretation of geoelectrical resistivity sounding
   measurements: Geophysical Prospecting, 19, 192--217;
   DOI: |_| `10.1111/j.1365-2478.1971.tb00593.x
   <http://doi.org/10.1111/j.1365-2478.1971.tb00593.x>`_.
.. [Haines_and_Jones_1988] Haines, G. V., and A. G. Jones, 1988, Logarithmic
   Fourier transformation: Geophysical Journal, 92, 171--178;
   DOI: |_| `10.1111/j.1365-246X.1988.tb01131.x
   <http://doi.org/10.1111/j.1365-246X.1988.tb01131.x>`_.
.. [Hamilton_2000] Hamilton, A. J. S., 2000, Uncorrelated modes of the
   non-linear power spectrum: Monthly Notices of the Royal Astronomical
   Society, 312, pages 257-284; DOI: |_| `10.1046/j.1365-8711.2000.03071.x
   <http://doi.org/10.1046/j.1365-8711.2000.03071.x>`_; Website of FFTLog:
   `casa.colorado.edu/~ajsh/FFTLog <http://casa.colorado.edu/~ajsh/FFTLog>`_.
.. [Hunziker_et_al_2015] Hunziker, J., J. Thorbecke, and E. Slob, 2015, The
   electromagnetic response in a layered vertical transverse isotropic medium:
   A new look at an old problem: Geophysics, 80(1), F1--F18;
   DOI: |_| `10.1190/geo2013-0411.1
   <http://doi.org/10.1190/geo2013-0411.1>`_;
   Software: `software.seg.org/2015/0001 <http://software.seg.org/2015/0001>`_.
.. [Key_2009] Key, K., 2009, 1D inversion of multicomponent, multifrequency
   marine CSEM data: Methodology and synthetic studies for resolving thin
   resistive layers: Geophysics, 74(2), F9--F20; DOI: |_| `10.1190/1.3058434
   <http://doi.org/10.1190/1.3058434>`_.
   Software: `marineemlab.ucsd.edu/Projects/Occam/1DCSEM
   <http://marineemlab.ucsd.edu/Projects/Occam/1DCSEM>`_.
.. [Key_2012] Key, K., 2012, Is the fast Hankel transform faster than
   quadrature?: Geophysics, 77(3), F21--F30; DOI: |_| `10.1190/geo2011-0237.1
   <http://doi.org/10.1190/geo2011-0237.1>`_;
   Software: `software.seg.org/2012/0003 <http://software.seg.org/2012/0003>`_.
.. [Kong_2007] Kong, F. N., 2007, Hankel transform filters for dipole antenna
   radiation in a conductive medium: Geophysical Prospecting, 55, 83--89;
   DOI: |_| `10.1111/j.1365-2478.2006.00585.x
   <http://doi.org/10.1111/j.1365-2478.2006.00585.x>`_.
.. [Shanks_1955] Shanks, D., 1955, Non-linear transformations of divergent and
   slowly convergent sequences: Journal of Mathematics and Physics, 34, 1--42;
   DOI: |_| `10.1002/sapm19553411
   <http://doi.org/10.1002/sapm19553411>`_.
.. [Slob_et_al_2010] Slob, E., J. Hunziker, and W. A. Mulder, 2010, Green's
   tensors for the diffusive electric field in a VTI half-space: PIER, 107,
   1--20: DOI: |_| `10.2528/PIER10052807
   <http://doi.org/10.2528/PIER10052807>`_.
.. [Talman_1978] Talman, J. D., 1978, Numerical Fourier and Bessel transforms
    in logarithmic variables: Journal of Computational Physics, 29, pages
    35-48; DOI: |_| `10.1016/0021-9991(78)90107-9
    <http://doi.org/10.1016/0021-9991(78)90107-9>`_.
.. [Trefethen_2000] Trefethen, L. N., 2000, Spectral methods in MATLAB: Society
   for Industrial and Applied Mathematics (SIAM), volume 10 of Software,
   Environments, and Tools, chapter 12, page 129;
   DOI: |_| `10.1137/1.9780898719598.ch12
   <http://doi.org/10.1137/1.9780898719598.ch12>`_.
.. [Weniger_1989] Weniger, E. J., 1989, Nonlinear sequence transformations for
   the acceleration of convergence and the summation of divergent series:
   Computer Physics Reports, 10, 189--371;
   arXiv: |_| `abs/math/0306302 <https://arxiv.org/abs/math/0306302>`_.
.. [Werthmuller_2017] Werthmüller, D., 2017, An open-source full 3D
   electromagnetic modeler for 1D VTI media in Python: empymod: Geophysics,
   82(6), WB9-WB19; DOI: |_| `10.1190/geo2016-0626.1
   <http://doi.org/10.1190/geo2016-0626.1>`_.
.. [Werthmuller_2017b] Werthmüller, D., 2017, Getting started with
   controlled-source electromagnetic 1D modeling: The Leading Edge, 36,
   352-355;
   DOI: |_| `10.1190/tle36040352.1 <http://doi.org/10.1190/tle36040352.1>`_.
.. [Wynn_1956] Wynn, P., 1956, On a device for computing the
   :math:`e_m(S_n)` tranformation: Math. Comput., 10, 91--96;
   DOI: |_| `10.1090/S0025-5718-1956-0084056-6
   <http://doi.org/10.1090/S0025-5718-1956-0084056-6>`_.
.. [Ziolkowski_and_Slob] Ziolkowski, A., and E. Slob, 2018, Introduction to
   Controlled-Source Electromagnetic Methods: Cambridge University Press;
   expected to be published late 2018.

"""
# Copyright 2016-2018 Dieter Werthmüller
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

# Import main modelling routines to make them available as primary functions
from .model import bipole, dipole, analytical
__all__ = ['bipole', 'dipole', 'analytical']

# Version
__version__ = '1.6.2'
