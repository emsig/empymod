Transforms
##########

Included **Hankel transforms**:

- Digital Linear Filters *DLF*
- Quadrature with Extrapolation *QWE*
- Adaptive quadrature *QUAD*

Included **Fourier transforms**:

- Digital Linear Filters *DLF*
- Quadrature with Extrapolation *QWE*
- Logarithmic Fast Fourier Transform *FFTLog*
- Fast Fourier Transform *FFT*


Digital Linear Filters
----------------------
The module ``empymod.filters`` comes with many DLFs for the Hankel and the
Fourier transform. If you want to export one of these filters to plain ascii
files you can use the ``tofile``-routine of each filter:

.. code-block:: python

    >>> import empymod
    >>> # Load a filter
    >>> filt = empymod.filters.wer_201_2018()
    >>> # Save it to pure ascii-files
    >>> filt.tofile()
    >>> # This will save the following three files:
    >>> #    ./filters/wer_201_2018_base.txt
    >>> #    ./filters/wer_201_2018_j0.txt
    >>> #    ./filters/wer_201_2018_j1.txt

Similarly, if you want to use an own filter you can do that as well. The filter
base and the filter coefficient have to be stored in separate files:

.. code-block:: python

    >>> import empymod
    >>> # Create an empty filter;
    >>> # Name has to be the base of the text files
    >>> filt = empymod.filters.DigitalFilter('my-filter')
    >>> # Load the ascii-files
    >>> filt.fromfile()
    >>> # This will load the following three files:
    >>> #    ./filters/my-filter_base.txt
    >>> #    ./filters/my-filter_j0.txt
    >>> #    ./filters/my-filter_j1.txt
    >>> # and store them in filt.base, filt.j0, and filt.j1.

The path can be adjusted by providing ``tofile`` and ``fromfile`` with a
``path``-argument.


FFTLog
------

FFTLog is the logarithmic analogue to the Fast Fourier Transform FFT originally
proposed by [Talm78]_. The code used by ``empymod`` was published in Appendix B
of [Hami00]_ and is publicly available at `casa.colorado.edu/~ajsh/FFTLog
<http://casa.colorado.edu/~ajsh/FFTLog>`_. From the ``FFTLog``-website:

*FFTLog is a set of fortran subroutines that compute the fast Fourier or Hankel
(= Fourier-Bessel) transform of a periodic sequence of logarithmically spaced
points.*

FFTlog can be used for the Hankel as well as for the Fourier Transform, but
currently ``empymod`` uses it only for the Fourier transform. It uses a
simplified version of the python implementation of FFTLog, ``pyfftlog``
(`github.com/prisae/pyfftlog <https://github.com/prisae/pyfftlog>`_).

[HaJo88]_ proposed a logarithmic Fourier transform (abbreviated by the authors
as LFT) for electromagnetic geophysics, also based on [Talm78]_. I do not know
if Hamilton was aware of the work by Haines and Jones. The two publications
share as reference only the original paper by Talman, and both cite a
publication of Anderson; Hamilton cites [Ande82]_, and Haines and Jones cite
[Ande79]_. Hamilton probably never heard of Haines and Jones, as he works in
astronomy, and Haines and Jones was published in the *Geophysical Journal*.

Logarithmic FFTs are not widely used in electromagnetics, as far as I know,
probably because of the ease, speed, and generally sufficient precision of the
digital filter methods with sine and cosine transforms ([Ande75]_). However,
comparisons show that FFTLog can be faster and more precise than digital
filters, specifically for responses with source and receiver at the interface
between air and subsurface. Credit to use FFTLog in electromagnetics goes to
David Taylor who, in the mid-2000s, implemented FFTLog into the forward
modellers of the company Multi-Transient ElectroMagnetic (MTEM Ltd, later
Petroleum Geo-Services PGS). The implementation was driven by land responses,
where FFTLog can be much more precise than the filter method for very early
times.


Notes on Fourier Transform
--------------------------

The Fourier transform to obtain the space-time domain impulse response from the
complex-valued space-frequency response can be calculated by either a
cosine transform with the real values, or a sine transform with the imaginary
part,

.. math::

    E(r, t)^\text{Impulse} &= \ \frac{2}{\pi}\int^\infty_0 \Re[E(r, \omega)]\
                        \cos(\omega t)\ \text{d}\omega \ , \\
            &= -\frac{2}{\pi}\int^\infty_0 \Im[E(r, \omega)]\
                \sin(\omega t)\ \text{d}\omega \ ,

see, e.g., [Ande75]_ or [Key12]_. Quadrature-with-extrapolation, FFTLog, and
obviously the sine/cosine-transform all make use of this split.

To obtain the step-on response the frequency-domain result is first divided
by :math:`\mathrm{i}\omega`, in the case of the step-off response it is
additionally multiplied by -1. The impulse-response is the time-derivative of
the step-response,

.. math::

    E(r, t)^\text{Impulse} =
                        \frac{\partial\ E(r, t)^\text{step}}{\partial t}\ .

Using :math:`\frac{\partial}{\partial t} \Leftrightarrow \mathrm{i}\omega` and
going the other way, from impulse to step, leads to the divison by
:math:`\mathrm{i}\omega`. (This only holds because we define in accordance with
the causality principle that :math:`E(r, t \le 0) = 0`).

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

    E(r, t)^\text{Step-on} = - \frac{2}{\pi}\int^\infty_0
                            \Im\left[\frac{E(r,\omega)}{\mathrm{i}
                            \omega}\right]\
                            \sin(\omega t)\ \text{d}\omega \ ,

and the step-off by

.. math::

    E(r, t)^\text{Step-off} = - \frac{2}{\pi}\int^\infty_0
                             \Re\left[\frac{E(r,\omega)}{\mathrm{i}
                             \omega}\right]\
                             \cos(\omega t)\ \text{d}\omega \ .


Laplace domain
--------------

It is also possible to calculate the response in the **Laplace domain**, by
using a real value for :math:`s` instead of the complex value
:math:`\mathrm{i}\omega``. This simplifies the problem from complex numbers to
real numbers. However, the transform from Laplace-to-time domain is not as
robust as the transform from frequency-to-time domain, and is currently not
implemented in ``empymod``. To calculate Laplace-domain responses instead
of frequency-domain responses simply provide negative frequency values. If all
provided frequencies :math:`f` are negative then :math:`s` is set to :math:`-f`
instead of the frequency-domain :math:`s=2\mathrm{i}\pi f`.
