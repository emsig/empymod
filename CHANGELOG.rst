Changelog
#########

v1.7.0 - *2018-05-23*
---------------------

Merge ``empyscripts`` into ``empymod`` under ``empymod.scripts``.

- Clear separation between mandatory and optional imports:

  - Mandatory:

    - ``numpy``
    - ``scipy``

  - Optional:

    - ``numexpr`` (for ``empymod.kernel``)
    - ``matplotlib`` (for ``empymod.scripts.fdesign``)
    - ``IPython`` (for ``empymod.scripts.printinfo``)

- Broaden namespace of ``empymod``. All public functions from the various
  modules and the modules from ``empymod.scripts`` are now available under
  ``empymod`` directly.


v1.6.2 - *2018-05-21*
---------------------

These changes should make calculations using ``QWE`` and ``QUAD`` for the
Hankel transform for cases which do not require all kernels faster; sometimes
as much as twice as fast. However, it might make calculations which do require
all kernels a tad slower, as more checks had to be included. (Related to
[`empymod#11 <https://github.com/empymod/empymod/issues/11>`_]; basically
including for ``QWE`` and ``QUAD`` what was included for ``DLF`` in version
1.6.0.)

- ``transform``:

  - ``dlf``:

    - Improved by avoiding unnecessary multiplications/summations for empty
      kernels and applying the angle factor only if it is not 1.
    - Empty/unused kernels can now be input as ``None``, e.g. ``signal=(PJ0,
      None, None)``.
    - ``factAng`` is new optional for the Hankel transform, as is ``ab``.

  - ``hqwe``: Avoids unnecessary calculations for zero kernels, improving speed
    for these cases.

  - ``hquad``, ``quad``: Avoids unnecessary calculations for zero kernels,
    improving speed for these cases.

- ``kernel``:

  - Simplify ``wavenumber``
  - Simplify ``angle_factor``


v1.6.1 - *2018-05-05*
---------------------

Secondary field calculation.

- Add the possibility to calculate secondary fields only (excluding the direct
  field) by passing the argument ``xdirect=None``. The complete
  ``xdirect``-signature is now (only affects calculation if src and rec are in
  the same layer):

  - If True, direct field is calculated analytically in the frequency domain.
  - If False, direct field is calculated in the wavenumber domain.
  - If None, direct field is excluded from the calculation, and only reflected
    fields are returned (secondary field).

- Bugfix in ``model.analytical`` for ``ab=[36, 63]`` (zeroes)
  [`empymod#16 <https://github.com/empymod/empymod/issues/16>`_].


v1.6.0 - *2018-05-01*
---------------------

This release is not completely backwards compatible for the main modelling
routines in ``empymod.model``, but almost. Read below to see which functions
are affected.

- Improved Hankel DLF
  [`empymod#11 <https://github.com/empymod/empymod/issues/11>`_].
  ``empymod.kernel.wavenumber`` always returns three kernels, ``PJ0``, ``PJ1``,
  and ``PJ0b``. The first one is angle-independent, the latter two depend on
  the angle. Now, depending of what source-receiver configuration is chosen,
  some of these might be zero. If-statements were now included to avoid the
  calculation of the DLF, interpolation, and reshaping for 0-kernels, which
  improves speed for these cases.

- Unified DLF arguments
  [`empymod#10 <https://github.com/empymod/empymod/issues/10>`_].

  These changes are backwards compatible for all main modelling routines in
  ``empymod.model``. However, they are not backwards compatible for the
  following routines:

  - ``empymod.model.fem`` (removed ``use_spline``),
  - ``empymod.transform.fht`` (removed ``use_spline``),
  - ``empymod.transform.hqwe`` (removed ``use_spline``),
  - ``empymod.transform.quad`` (removed ``use_spline``),
  - ``empymod.transform.dlf`` (``lagged``, ``splined`` => ``pts_per_dec``),
  - ``empymod.utils.check_opt`` (no longer returns ``use_spline``),
  - ``empymod.utils.check_hankel`` (changes in ``pts_per_dec``), and
  - ``empymod.utils.check_time`` (changes in ``pts_per_dec``).

  The function ``empymod.utils.spline_backwards_hankel`` can be used for
  backwards compatibility.

  Now the Hankel and Fourier DLF have the same behaviour for ``pts_per_dec``:

  - ``pts_per_dec = 0``: Standard DLF,
  - ``pts_per_dec < 0``: Lagged Convolution DLF, and
  - ``pts_per_dec > 0``: Splined DLF.

  **There is one exception** which is not backwards compatible: Before, if
  ``opt=None`` and ``htarg={pts_per_dec: != 0}``, the ``pts_per_dec`` was not
  used for the FHT and the QWE. New, this will be used according to the above
  definitions.

- Bugfix in ``model.wavenumber`` for ``ab=[36, 63]`` (zeroes).


v1.5.2 - *2018-04-25*
---------------------

- DLF improvements:

  - Digital linear filter (DLF) method for the Fourier transform can now be
    carried out without spline, providing 0 for ``pts_per_dec`` (or any
    integer smaller than 1).

  - Combine kernel from ``fht`` and ``ffht`` into ``dlf``, hence separate DLF
    from other calculations, as is done with QWE (``qwe`` for ``hqwe`` and
    ``fqwe``).

  - Bug fix regarding ``transform.get_spline_values``; a DLF with
    ``pts_per_dec`` can now be shorter then the corresponding filter.


v1.5.1 - *2018-02-24*
---------------------

- Documentation:

  - Simplifications: avoid duplication as much as possible between the website
    (`empymod.github.io <https://empymod.github.io>`_), the manual
    (`empymod.readthedocs.io <https://empymod.readthedocs.io>`_), and the
    ``README`` (`github.com/empymod/empymod
    <https://github.com/empymod/empymod>`_).

    - Website has now only *Features* and *Installation* in full, all other
      information comes in the form of links.
    - ``README`` has only information in the form of links.
    - Manual contains the ``README``, and is basically the main document for
      all information.

  - Improvements: Change some remaining ``md``-syntax to ``rst``-syntax.

  - FHT -> DLF: replace FHT as much as possible, without breaking backwards
    compatibility.


v1.5.0 - *2018-01-02*
---------------------

- Minimum parameter values can now be set and verified with
  ``utils.set_minimum`` and ``utils.get_minimum``.

- New Hankel filter ``wer_201_2018``.

- ``opt=parallel`` has no effect if ``numexpr`` is not built against Intel's
  VML. (Use ``import numexpr; numexpr.use_vml`` to see if your ``numexpr`` uses
  VML.)

- Bug fixes

- Version of manuscript submission to geophysics for the DLF article.


v1.4.4 - *2017-09-18*
---------------------

[This was meant to be 1.4.3, but due to a setup/pypi/anaconda-issue I had to
push it to 1.4.4; so there isn't really a version 1.4.3.]

- Add TE/TM split to diffusive ee-halfspace solution.

- Improve ``kernel.wavenumber`` for fullspaces.

- Extended ``fQWE`` and ``fftlog`` to be able to use the cosine-transform. Now
  the cosine-transform with the real-part frequency response is used internally
  if a switch-off response (``signal=-1``) is required, rather than calculating
  the switch-on response (with sine-transform and imaginary-part frequency
  response) and subtracting it from the DC value.

- Bug fixes

- Version of CSEM book.


v1.4.2 - *2017-06-04*
---------------------

- Bugfix: Fixed squeeze in ``model.analytical`` with ``solution='dsplit'``.

- Version of final submission of manuscript to Geophysics.


v1.4.1 - *2017-05-30*
---------------------

[This was meant to be 1.4.0, but due to a setup/pypi/anaconda-issue I had to
push it to 1.4.1; so there isn't really a version 1.4.0.]

- New home: `empymod.github.io <https://empymod.github.io>`_ as entry point,
  and the project page on `github.com/empymod <https://github.com/empymod>`_.
  All empymod-repos moved to the new home.

  - /prisae/empymod -> /empymod/empymod
  - /prisae/empymod-notebooks -> /empymod/example-notebooks
  - /prisae/empymod-geo2017 -> /empymod/article-geo2017
  - /prisae/empymod-tle2017 -> /empymod/article-tle2017

- Modelling routines:

  - New modelling routine ``model.analytical``, which serves as a front-end to
    ``kernel.fullspace`` or ``kernel.halfspace``.
  - Remove legacy routines ``model.time`` and ``model.frequency``.  They are
    covered perfectly by ``model.dipole``.
  - Improved switch-off response (calculate and subtract from DC).
  - ``xdirect`` adjustments:

    - ``isfullspace`` now respects ``xdirect``.
    - Removed ``xdirect`` from ``model.wavenumber`` (set to ``False``).

- Kernel:

  - Modify ``kernel.halfspace`` to use same input as other kernel functions.
  - Include time-domain ee halfspace solution into ``kernel.halfspace``;
    possible to obtain direct, reflected, and airwave separately, as well as
    only fullspace solution (all for the diffusive approximation).


v1.3.0 - *2017-03-30*
---------------------

- Add additional transforms and improve QWE:

  - Conventional adaptive quadrature (QUADPACK) for the Hankel transform;
  - Conventional FFT for the Fourier transform.
  - Add ``diff_quad`` to ``htarg``/``ftarg`` of QWE, a switch parameter for
    QWE/QUAD.
  - Change QWE/QUAD switch from comparing first interval to comparing all
    intervals.
  - Add parameters for QUAD (a, b, limit) into ``htarg``/``ftarg`` for QWE.

- Allow ``htarg``/``ftarg`` as dict additionally to list/tuple.

- Improve ``model.gpr``.

- Internal changes:

  - Rename internally the sine/cosine filter from ``fft`` to ``ffht``, because
    of the addition of the Fast Fourier Transform ``fft``.

- Clean-up repository

  - Move ``notebooks`` to /prisae/empymod-notebooks
  - Move ``publications/Geophysics2017`` to /prisae/empymod-geo2017
  - Move ``publications/TheLeadingEdge2017`` to /prisae/empymod-tle2017

- Bug fixes and documentation improvements


v1.2.1 - *2017-03-11*
---------------------

- Change default filter from ``key_401_2009`` to ``key_201_2009`` (because of
  warning regarding 401 pt filter in source code of ``DIPOLE1D``.)

- Since 06/02/2017 installable via pip/conda.

- Bug fixes


v1.2.0 - *2017-02-02*
---------------------

- New routine:

  - General modelling routine ``bipole`` (replaces ``srcbipole``): Model the EM
    field for arbitrarily oriented, finite length bipole sources and receivers.

- Added a test suite:

  - Unit-tests of small functions.
  - Framework-tests of the bigger functions:

    - Comparing to status quo (regression tests),
    - Comparing to known analytical solutions,
    - Comparing different options to each other,
    - Comparing to other 1D modellers (EMmod, DIPOLE1D, GREEN3D).

  - Incorporated with Travis CI and Coveralls.

- Internal changes:

  - Add kernel count (printed if verb > 1).
  - ``numexpr`` is now only required if ``opt=='parallel'``. If ``numexpr`` is
    not found, ``opt`` is reset to ``None`` and a warning is printed.
  - Cleaned-up wavenumber-domain routine.
  - theta/phi -> azimuth/dip; easier to understand.
  - Refined verbosity levels.
  - Lots of changes in ``utils``, with regards to the new routine ``bipole``
    and with regards to verbosity. Moved all warnings out from ``transform``
    and ``model`` into ``utils``.

- Bug fixes


v1.1.0 - *2016-12-22*
---------------------

- New routines:

  - New ``srcbipole`` modelling routine: Model an arbitrarily oriented, finite
    length bipole source.
  - Merge ``frequency`` and ``time`` into ``dipole``. (``frequency`` and
    ``time`` are still available.)
  - ``dipole`` now supports multiple sources.

- Internal changes:

  - Replace ``get_Gauss_Weights`` with ``scipy.special.p_roots``
  - ``jv(0,x)``, ``jv(1,x)`` -> ``j0(x)``, ``j1(x)``
  - Replace ``param_shape`` in ``utils`` with ``_check_var`` and
    ``_check_shape``.
  - Replace ``xco`` and ``yco`` by ``angle`` in ``kernel.fullspace``
  - Replace ``fftlog`` with python version.
  - Additional sine-/cosine-filters: ``key_81_CosSin_2009``,
    ``key_241_CosSin_2009``, and ``key_601_CosSin_2009``.

- Bug fixes


v1.0.0 - *2016-11-29*
---------------------

- Initial release; state of manuscript submission to geophysics.
