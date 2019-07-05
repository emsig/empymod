Changelog
#########


v1.8.3 - *2019-07-05*
---------------------

- Use ``scooby`` for ``Versions`` (printinfo), change name to ``Report``.
- DOC: Correct return statement if ``mrec=True``.
- Typos and correct links for new asv/bench.
- Bump requirement to SciPy>=1.0.0, remove warning regarding memory leak in
  SciPy 0.19.0.


v1.8.2 - *2019-04-26*
---------------------

- ``pts_per_dec`` are now floats, not integers, which gives more flexibility.
- Bugfix: ``pts_per_dec`` for DLF was actually points per ``e``, not per
  decade, as the natural logarithm was used.
- New ``Versions``-class; improvement over the ``versions``-function, as it
  automatically detects if it can print html or not.
- Maintenance: Update ``np.load`` in tests with ``allow_pickle=True`` for
  changes in numpy v1.16.3.
- Lots of changes to accommodate ``emg3d`` within the ``empymod``-org:

  - Adjust website, move stuff from website into README.md.
  - /empymod/example-notebooks -> /empymod/empymod-examples.
  - /empymod/asv -> /empymod/empymod-asv (and therefore now available at
    `empymod.github.io/empymod-asv <https://empymod.github.io/empymod-asv>`_).
  - /empymod/bench -> /empymod/empymod-bench.

- Move manual from ``empymod/__init__.py`` to the ``docs/manual.rst``, and the
  references to its own file. Change reference style.
- Move credits for initial funding from the license-section of the manual to
  CREDITS.rst, where it belongs.


v1.8.1 - *2018-11-20*
---------------------

- Many little improvements in the documentation.
- Some code improvements through the use of codacy.
- Remove testing of Python 3.4; officially supported are now Python 3.5-3.7.
- Version of the `filter article <https://github.com/empymod/article-fdesign>`_
  (DLF) in geophysics and of the `CSEM book
  <https://github.com/empymod/csem-ziolkowski-and-slob>`_.


v1.8.0 - *2018-10-26*
---------------------

- ``model.bipole``, ``model.dipole``, and ``model.analytical`` have now a hook
  which users can exploit to insert their own calculation of ``etaH``,
  ``etaV``, ``zetaH``, and ``zetaV``. This can be used, for instance, to model
  a Cole-Cole IP survey. See the manual or the example-notebooks for more
  information.

- ``model.wavenumber`` renamed to ``model.dipole_k`` to avoid name clash with
  ``kernel.wavenumber``. For now ``model.wavenumber`` continues to exist, but
  raises a depreciation warning.

- ``xdirect`` default value changed from ``True`` to ``False``.

- Possibility to provide interpolated points (``int_pts``) to
  ``transform.dlf``.

The following changes are backwards incompatible if you directly used
``transform.fht``, ``transform.hqwe``, or ``transform.hquad``. Nothing changes
for the user-facing routines in ``model``:

- ``empymod.fem`` now passes ``factAng`` to ``empymod.transform``, not
  ``angle``; this saves some time if looped over offsets or frequencies, as it
  is not repeatedly calculated within ``empymod.transform``.

- Use ``get_spline_values`` in ``empymod.fem`` for Hankel DLF, instead of in
  ``empymod.fht``. Gives a speed-up if looped over offsets or frequencies.
  Should be in ``utils``, but that would be heavily backwards incompatible.
  Move there in version 2.0.


v1.7.3 - *2018-07-16*
---------------------

- Small improvements related to speed as a result of the benchmarks introduced
  in v1.7.2:

  - Kernels which do not exist for a given ``ab`` are now returned as ``None``
    from ``kernel.wavenumber`` instead of arrays of zeroes. This permits for
    some time saving in the transforms. This change is backwards incompatible
    if you directly used ``kernel.wavenumber``. Nothing changes for the
    user-facing routines in ``model``.

  - Adjustments in ``transform`` with regard to the ``None`` returned by
    ``kernel.wavenumber``. The kernels are not checked anymore if they are all
    zeroes (which can be slow for big arrays). If they are not None, they will
    be processed.

  - Various small improvements for speed to ``transform.dlf`` (i.e.
    ``factAng``; ``log10``/``log``; re-arranging).


v1.7.2 - *2018-07-07*
---------------------

- Benchmarks: ``empymod`` has now a benchmark suite, see `empymod/asv
  <https://github.com/empymod/asv>`_.

- Fixed a bug in ``bipole`` for time-domain responses with several receivers or
  sources with different depths. (Simply failed, as wrong dimension was
  provided to ``tem``).

- Small improvements:

  - Various simplifications or cleaning of the code base.
  - Small change (for speed) in check if kernels are empty in ``transform.dlf``
    and ``transform.qwe``.


v1.7.1 - *2018-06-19*
---------------------

- New routines in ``empymod.filters.DigitalFilter``: Filters can now be saved
  to or loaded from pure ascii-files.

- Filters and inversion result from ``empymod.scripts.fdesign`` are now by
  default saved in plain text. The filters with their internal routine, the
  inversion result with ``np.savetxt``. Compressed saving can be achieved by
  giving a name with a '.gz'-ending.

- Change in ``empymod.utils``:

  - Renamed ``_min_param`` to ``_min_res``.
  - Anisotropy ``aniso`` is no longer directly checked for its minimum value.
    Instead, res*aniso**2, hence vertical resistivity, is checked with
    ``_min_res``, and anisotropy is subsequently re-calculated from it.
  - The parameters ``epermH``, ``epermV``, ``mpermH``, and ``mpermV`` can now
    be set to 0 (or any positive value) and do not depend on ``_min_param``.

- ``printinfo``: Generally improved; prints now MKL-info (if available)
  independently of ``numexpr``.

- Simplification of ``kernel.reflections`` through re-arranging.

- Bug fixes

- Version of re-submission of the DLF article to geophysics.


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
