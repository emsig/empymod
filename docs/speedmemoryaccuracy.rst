Speed, memory, and accuracy
###########################

There is the usual trade-off between speed, memory, and accuracy. Very
generally speaking we can say that the *DLF* is faster than *QWE*, but *QWE* is
much easier on memory usage. *QWE* allows you to control the accuracy. A
standard quadrature in the form of *QUAD* is also provided. *QUAD* is generally
orders of magnitudes slower, and more fragile depending on the input arguments.
However, it can provide accurate results where *DLF* and *QWE* fail.

Parts of the kernel can run in parallel using ``numexpr``. This option is
activated by setting ``opt='parallel'`` (see subsection :ref:`Parallelisation
<parallelisation>`). It is switched off by default.


Memory
------
By default ``empymod`` will try to carry out the calculation in one go, without
looping. If your model has many offsets and many frequencies this can be heavy
on memory usage. Even more so if you are calculating time-domain responses for
many times. If you are running out of memory, you should use either
``loop='off'`` or ``loop='freq'`` to loop over offsets or frequencies,
respectively. Use ``verb=3`` to see how many offsets and how many frequencies
are calculated internally.



Depths, Rotation, and Bipole
----------------------------
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


.. _parallelisation:

Parallelisation
---------------
If ``opt = 'parallel'``, six (*) of the most time-consuming statements are
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

This parallelisation will make ``empymod`` faster (by using more threads) if
you calculate a lot of offsets/frequencies at once, but slower for few
offsets/frequencies.

(*) These statements are (following the notation of [HuTS15]_): :math:`\Gamma`
(below eq. 19); :math:`W^{u, d}_n` (eq. 74), :math:`r^\pm_n` (eq. 65);
:math:`R^\pm_n` (eq. 64); :math:`P^{u, d; \pm}_s` (eq. 81); :math:`M_s` (eq.
82), and their corresponding bar-ed versions provided in the appendix (e.g.
:math:`\bar{\Gamma}`). In big models, more than 95 % of the calculation is
spent in the calculation of these six equations, and most of the time therefore
in ``np.sqrt`` and ``np.exp``, or generally in ``numpy``-``ufuncs`` which are
implemented and executed in compiled C-code. For smaller models or if
transforms with interpolations are used then all the other parts also start to
play a role. However, those models generally execute comparably fast.


Lagged Convolution and Splined Transforms
-----------------------------------------
Both Hankel and Fourier DLF have three options, which can be controlled via
the ``htarg['pts_per_dec']`` and ``ftarg['pts_per_dec']`` parameters:

    - ``pts_per_dec=0`` : *Standard DLF*;
    - ``pts_per_dec<0`` : *Lagged Convolution DLF*: Spacing defined by filter
      base, interpolation is carried out in the input domain;
    - ``pts_per_dec>0`` : *Splined DLF*: Spacing defined by ``pts_per_dec``,
      interpolation is carried out in the output domain.

Similarly, interpolation can be used for ``QWE`` by setting ``pts_per_dec`` to
a value bigger than 0.

The Lagged Convolution and Splined options should be used with caution, as they
use interpolation and are therefore less precise than the standard version.
However, they can significantly speed up *QWE*, and massively speed up *DLF*.
Additionally, the interpolated versions minimizes memory requirements a lot.
Speed-up is greater if all source-receiver angles are identical. Note that
setting ``pts_per_dec`` to something else than 0 to calculate only one offset
(Hankel) or only one time (Fourier) will be slower than using the standard
version. Similarly, the standard version is usually the fastest when using the
``parallel`` option (``numexpr``).

*QWE*: Good speed-up is also achieved for *QWE* by setting ``maxint`` as low as
possible. Also, the higher ``nquad`` is, the higher the speed-up will be.

*DLF*: Big improvements are achieved for long DLF-filters and for many
offsets/frequencies (thousands).

.. warning::

    Keep in mind that setting ``pts_per_dec`` to something else than 0 uses
    interpolation, and is therefore not as accurate as the standard version.
    Use with caution and always compare with the standard version to verify
    if you can apply interpolation to your problem at hand!

Be aware that *QUAD* (Hankel transform) *always* use the splined version and
*always* loops over offsets. The Fourier transforms *FFTlog*, *QWE*, and *FFT*
always use interpolation too, either in the frequency or in the time domain.
With the *DLF* Fourier transform (sine and cosine transforms) you can choose
between no interpolation and interpolation (splined or lagged).

The splined versions of *QWE* check whether the ratio of any two adjacent
intervals is above a certain threshold (steep end of the wavenumber or
frequency spectrum). If it is, it carries out *QUAD* for this interval instead
of *QWE*. The threshold is stored in ``diff_quad``, which can be changed within
the parameter ``htarg`` and ``ftarg``.

For a graphical explanation of the differences between standard DLF, lagged
convolution DLF, and splined DLF for the Hankel and the Fourier transforms see
the example in the Gallery.

Looping
-------
By default, you can calculate many offsets and many frequencies
all in one go, vectorized (for the *DLF*), which is the default. The ``loop``
parameter gives you the possibility to force looping over frequencies or
offsets. This parameter can have severe effects on both runtime and memory
usage. Play around with this factor to find the fastest version for your
problem at hand. It ALWAYS loops over frequencies if ``ht = 'QWE'/'QUAD'`` or
if ``ht = 'FHT'`` and ``pts_per_dec!=0`` (Lagged Convolution or Splined Hankel
DLF). All vectorized is very fast if there are few offsets or few frequencies.
If there are many offsets and many frequencies, looping over the smaller of the
two will be faster. Choosing the right looping together with ``opt =
'parallel'`` can have a huge influence.


Vertical components and ``xdirect``
-----------------------------------
Calculating the direct field in the wavenumber-frequency domain
(``xdirect=False``; the default) is generally faster than calculating it in the
frequency-space domain (``xdirect=True``).

However, using ``xdirect = True`` can improve the result (if source and
receiver are in the same layer) to calculate:

    - the vertical electric field due to a vertical electric source,
    - configurations that involve vertical magnetic components (source or
      receiver),
    - all configurations when source and receiver depth are exactly the same.

The Hankel transforms methods are having sometimes difficulties transforming
these functions.


Time-domain land CSEM
---------------------
The derivation, as it stands, has a near-singular behaviour in the
wavenumber-frequency domain when :math:`\kappa^2 = \omega^2\epsilon\mu`. This
can be a problem for land-domain CSEM calculations if source and receiver are
located at the surface between air and subsurface. Because most transforms do
not sample the wavenumber-frequency domain sufficiently to catch this
near-singular behaviour (hence not smooth), which then creates noise at early
times where the signal should be zero. To avoid the issue simply set
``epermH[0] = epermV[0] = 0``, hence the relative electric permittivity of the
air to zero. This trick obviously uses the diffusive approximation for the
air-layer, it therefore will not work for very high frequencies (e.g., GPR
calculations).

This trick works fine for all horizontal components, but not so much for the
vertical component. But then it is not feasible to have a vertical source or
receiver *exactly* at the surface. A few tips for these cases: The receiver can
be put pretty close to the surface (a few millimeters), but the source has to
be put down a meter or two, more for the case of vertical source AND receiver,
less for vertical source OR receiver. The results are generally better if the
source is put deeper than the receiver. In either case, the best is to first
test the survey layout against the analytical result (using
``empymod.analytical`` with ``solution='dhs'``) for a half-space, and
subsequently model more complex cases.

A common alternative to this trick is to apply a lowpass filter to filter out
the unstable high frequencies.
