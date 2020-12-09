Changelog
#########


Version 2
~~~~~~~~~


v2.0.x
""""""


v2.0.4: Move to EMSiG
---------------------

**2020-12-09**

Small maintenance release:

- Update github.com/empymod to github.com/emsig.
- Moved from Travis CI to GitHub Actions.


v2.0.3: Docs and gallery
------------------------

**2020-09-22**

- Documentation:

  - New section under *Tips and tricks* regarding *Zero horizontal offset*.

- Example gallery:

  - Re-organization of the section *Reproducing*: split *CSEM* into the
    two examples, rename all.
  - New example *Hunziker et al., 2015*, in the section *Reproducing*.
  - Update and maintain all of them.

- Maintenance:

  - Take care of deprecation warnings:

    - numpy: https://numpy.org/neps/nep-0034-infer-dtype-is-object.html
    - matplotlib: https://github.com/matplotlib/matplotlib/pull/16258

  - Correct docs re ``htarg`` for ``ht='quad'`` (``a``/``b`` vs
    ``lmin``/``lmax``).

  - Sphinx: Ensure ``_html_repr_`` is captured by RTD and other small
    improvements.

  - Chain errors.

  - Improve NumPy types.


v2.0.2: Fix example: ``tem_walktem``
------------------------------------

**2020-07-02**

- Fix example ``tem_walktem``, related to changes in ``scipy.quadrature``:
  Replace not-existing private name
  ``scipy.integrate.quadrature._cached_roots_legendre`` with the public name
  ``scipy.special.roots_legendre``.
- As a consequence of the above, changed in ``empymod`` the used, old name
  ``scipy.special.p_roots`` by new, more descriptive name
  ``scipy.special.roots_legendre``.
- Improve *publications*-section in the example gallery.
- Change error reporting to native (instead of ``print(bla)`` and ``raise
  Error`` do ``raise Error(bla)``); improve corresponding error testing by
  checking the error message too.


v2.0.1: Bug fixes: ftarg, docs, CI, req
---------------------------------------

**2020-06-19**

- Bugfix that using ``ftarg`` returned from ``utils.check_time`` as input for
  the same ``utils.check_time`` does not throw a warning in the case of
  ``fftlog`` and ``qwe``.
- Various micro-improvements and simplifications with regards to the
  documentation, testing, and requirement specifications.


v2.0.0: Numba
-------------

**2020-04-29**

This version is backwards incompatible and requires Python 3.6+.

- Numba:

  - Using ``numexpr`` is no longer a possibility. Instead, ``numba`` is a new
    dependency. All four kernel routines (``wavenumber``, ``greenfct``,
    ``reflections``, and ``fields``) are now numba-jitted functions.

- Removed:

  - Removed all deprecated functions.
  - Dropped support for Python 3.5; moved to f-strings.
  - Dropped testing for channel conda-forge. The problems encountered at the
    early development cycle of empymod with conda-forge do not exist any
    longer.

- New defaults:

  - ``EMArray``: ``.amp`` and ``.pha`` are now methods, not properties. Phase
    takes three optional boolean parameters ``deg=False``, ``unwrap=True``, and
    ``lag=True``, to get radians or degrees; unwrapped or not; and lag or lead
    defined phases.
  - The parameters ``epermV`` and ``mpermV`` are set to the values of
    ``epermH`` and ``mpermH``, respectively, if not provided (hence assuming
    isotropic behaviour). Before they were set to ones if not provided.

- Renaming:

  - ``transform.fht`` -> ``transform.hankel_dlf``
  - ``transform.hqwe`` -> ``transform.hankel_qwe``
  - ``transform.hquad`` -> ``transform.hankel_quad``
  - ``transform.ffht`` -> ``transform.fourier_dlf``
  - ``transform.fqwe`` -> ``transform.fourier_qwe``
  - ``transform.fftlog`` -> ``transform.fourier_fftlog``
  - ``transform.fft`` -> ``transform.fourier_fft``
  - ``transform.fhti`` -> ``transform.get_fftlog_input``
  - ``transform.get_spline_values`` -> ``transform.get_dlf_points``.
  - ``factAng`` -> ``ang_fact``
  - In ``htarg``-dict: ``fftfilt``-> ``dlf`` (filter name for Hankel-DLF)
  - In ``ftarg``-dict: ``fhtfilt``-> ``dlf`` (filter name for Fourier-DLF)
  - In ``ftarg``-dict: ``ft``-> ``kind`` (method in Fourier-DLF [sine/cosine])
  - Only dictionaries allowed for ``htarg`` and ``ftarg``; strings, lists, or
    tuples are not allowed any longer. They are also dictionaries internally
    now.
  - ``ht``: There is only one unique name for each method:  'dlf', 'qwe',
    'quad'.
  - ``ft``: There is only one unique name for each method:  'dlf', 'qwe',
    'fftlog', 'fft'.
  - Within ``transform``, change ``fhtarg``, ``qweargs``, and ``quadargs`` to
    ``htarg``; ``qweargs`` to ``ftarg``.

- Other changes:

  - All settings (``xdirect``, ``ht``, ``htarg``, ``ft``, ``ftarg``, ``loop``,
    ``verb``) are now extracted from ``kwargs``. This makes it possible that
    all ``model``-functions take the same keyword-arguments; warnings are
    raised if a particular parameter is not used in this function, but it
    doesn't fail (it fails, however, for unknown parameters). Pure positional
    calls including those parameters will therefore not work any longer.
  - Undo a change introduced in v1.8.0: ``get_dlf_points`` is calculated
    directly within ``transform.fht`` [`empymod#26
    <https://github.com/emsig/empymod/issues/26>`_].
  - Ensured that source and receiver inputs are not altered.
  - Significantly reduced top namespace; only functions from ``model`` are
    loaded into the top namespace now.


Version 1
~~~~~~~~~


v1.10.x
"""""""

v1.10.6: Various azimuths and dips at same depth
------------------------------------------------

**2020-03-04**

- ``empymod.bipole``

  - In the source and receiver format ``[x, y, z, azimuth, dip]``, azimuth and
    dip can now be either single values, or the same number as the other
    coordinates.
  - Bugfix (in ``utils.get_abs``): When different orientations were used
    exactly along the principal axes, at the same depth, only the first source
    was calculated [`empymod#74
    <https://github.com/emsig/empymod/issues/74>`_].


v1.10.5: Continuously in- or decreasing
---------------------------------------

**2020-02-21**

This is a small appendix to v1.10.4: Depths can now be defined in increasing or
decreasing order, as long as they are consistent. Model parameters have to be
defined in the same order. Hence all these are possible:

  - ``[-100, 0, 1000, 1050]`` -> left-handed system, low-to-high
  - ``[100, 0, -1000, -1050]`` -> right-handed system, high-to-low
  - ``[1050, 1000, 0, -100]`` -> left-handed system, high-to-low
  - ``[-1050, -1000, 0, 100]`` -> right-handed system, low-to-high


v1.10.4: Positive z down- or upwards
------------------------------------

**2020-02-16**

- New examples:

  - ``empymod`` can handle positive z down- or upwards (left-handed or
    right-handed coordinate systems; it was always possible, but not known nor
    documented). Adjusted documentation, docstrings, and added an example.
  - Example how to calculate the responses for the WalkTEM system.

- Minor things and bug fixes:

  - Change from relative to absolute imports.
  - Simplified releasing (no badges).
  - Python 3.8 is tested.
  - Fix: numpy now throws an error if the third argument of ``logspace`` is not
    an ``int``, some casting was therefore necessary within the code.


v1.10.3: Sphinx Gallery
-----------------------

**2019-11-11**

- Move examples to an integrated Sphinx-Gallery, generated each time.
- Move from conda-channel ``prisae`` to ``conda-forge``.
- Automatic deploy for PyPi and conda-forge.


v1.10.2: Always EMArray
-----------------------

**2019-11-06**

- Simplified and improved ``empymod.utils.EMArray``. Now every returned array
  from the main modelling routines ``bipole``, ``dipole``, ``loop``, and
  ``analytical`` is an EMArray with ``.amp``- and ``.pha``-attributes.
- Theme and documentation reworked, to be more streamlined with ``emg3d`` (for
  easier long-term maintenance).
- Travis now checks all the url's in the documentation, so there should be no
  broken links down the road. (Check is allowed to fail, it is visual QC.)
- Fixes to the ``setuptools_scm``-implementation (``MANIFEST.in``).
- ``ROADMAP.rst`` moved to GitHub-Projects; ``MAINTENANCE.rst`` included in
  manual.


v1.10.1: setuptools_scm
-----------------------

**2019-10-22**

- Typos from v1.10.0; update example in ``model.loop``.
- Implement ``setuptools_scm`` for versioning (adds git hashes for
  dev-versions).


v1.10.0: Loop source and receiver
---------------------------------

**2019-10-15**

- New modelling routine ``model.loop`` to model the electromagnetic frequency-
  or time-domain field due to an arbitrary rotated, magnetic source consisting
  of an electric loop, measured by arbitrary rotated, finite electric or
  magnetic bipole receivers or arbitrary rotated magnetic receivers consisting
  of electric loops.
- Move copyright from «Dieter Werthmüller» to «The empymod Developers», to be
  more inclusive and open the project for new contributors.


v1.9.x
"""""""

v1.9.0 : Laplace
----------------

**2019-10-04**

- Laplace-domain calculation: By providing a negative ``freq``-value, the
  calculation is carried out in the real Laplace domain ``s = freq`` instead of
  the complex frequency domain ``s = 2i*pi*freq``.
- Improvements to filter design and handling:

  - ``DigitalFilter`` now takes an argument (list of strings) for additional
    coefficients to the default ``j0``, ``j1``, ``sin``, and ``cos``.
  - ``fdesign`` can now be used with any name as attribute you want to describe
    the transform pair (until now it had to be either ``j0``, ``j1``, ``j2``,
    ``sin``, or ``cos``).
  - The provided sine and cosine transform pairs in ``fdesign`` can now be
    asked to return the inverse pair (time to frequency).

- Other tiny improvements and bug fixes.


v1.8.x
""""""


v1.8.3 : Scooby
---------------

**2019-07-05**

- Use ``scooby`` for ``Versions`` (printinfo), change name to ``Report``.
- DOC: Correct return statement if ``mrec=True``.
- Typos and correct links for new asv/bench.
- Bump requirement to SciPy>=1.0.0, remove warning regarding memory leak in
  SciPy 0.19.0.


v1.8.2 : pts_per_dec for DLF are now floats
-------------------------------------------

**2019-04-26**

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
    empymod.github.io/empymod-asv).
  - /empymod/bench -> /empymod/empymod-bench.

- Move manual from ``empymod/__init__.py`` to the ``docs/manual.rst``, and the
  references to its own file. Change reference style.
- Move credits for initial funding from the license-section of the manual to
  CREDITS.rst, where it belongs.


v1.8.1 : Version of Filter-article and CSEM-book
------------------------------------------------

**2018-11-20**

- Many little improvements in the documentation.
- Some code improvements through the use of codacy.
- Remove testing of Python 3.4; officially supported are now Python 3.5-3.7.
- Version of the `filter article <https://github.com/emsig/article-fdesign>`_
  (DLF) in geophysics and of the `CSEM book
  <https://github.com/emsig/csem-ziolkowski-and-slob>`_.


v1.8.0 : Hook for Cole-Cole IP and similar
------------------------------------------

**2018-10-26**

- ``model.bipole``, ``model.dipole``, and ``model.analytical`` have now a hook
  which users can exploit to insert their own calculation of ``etaH``,
  ``etaV``, ``zetaH``, and ``zetaV``. This can be used, for instance, to model
  a Cole-Cole IP survey. See the manual or the example-notebooks for more
  information.

- ``model.wavenumber`` renamed to ``model.dipole_k`` to avoid name clash with
  ``kernel.wavenumber``. For now ``model.wavenumber`` continues to exist, but
  raises a deprecation warning.

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


v1.7.x
""""""


v1.7.3 : Speed improvements following benchmarks
------------------------------------------------

**2018-07-16**

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


v1.7.2 : Benchmarked with asv
-----------------------------

**2018-07-07**

- Benchmarks: ``empymod`` has now a benchmark suite, see `emsig/empymod-asv
  <https://github.com/emsig/empymod-asv>`_.

- Fixed a bug in ``bipole`` for time-domain responses with several receivers or
  sources with different depths. (Simply failed, as wrong dimension was
  provided to ``tem``).

- Small improvements:

  - Various simplifications or cleaning of the code base.
  - Small change (for speed) in check if kernels are empty in ``transform.dlf``
    and ``transform.qwe``.


v1.7.1 : Load/save filters in plain text
----------------------------------------

**2018-06-19**

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


v1.7.0 : Move empyscripts into empymod.scripts
----------------------------------------------

**2018-05-23**

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


v1.6.x
""""""


v1.6.2 : Speed improvements for QUAD/QWE
----------------------------------------

**2018-05-21**

These changes should make calculations using ``QWE`` and ``QUAD`` for the
Hankel transform for cases which do not require all kernels faster; sometimes
as much as twice as fast. However, it might make calculations which do require
all kernels a tad slower, as more checks had to be included. (Related to
[`empymod#11 <https://github.com/emsig/empymod/issues/11>`_]; basically
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


v1.6.1 : Primary/secondary field
--------------------------------

**2018-05-05**

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
  [`empymod#16 <https://github.com/emsig/empymod/issues/16>`_].


v1.6.0 : More DLF improvements
------------------------------

**2018-05-01**

This release is not completely backwards compatible for the main modelling
routines in ``empymod.model``, but almost. Read below to see which functions
are affected.

- Improved Hankel DLF
  [`empymod#11 <https://github.com/emsig/empymod/issues/11>`_].
  ``empymod.kernel.wavenumber`` always returns three kernels, ``PJ0``, ``PJ1``,
  and ``PJ0b``. The first one is angle-independent, the latter two depend on
  the angle. Now, depending of what source-receiver configuration is chosen,
  some of these might be zero. If-statements were now included to avoid the
  calculation of the DLF, interpolation, and reshaping for 0-kernels, which
  improves speed for these cases.

- Unified DLF arguments
  [`empymod#10 <https://github.com/emsig/empymod/issues/10>`_].

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


v1.5.x
""""""


v1.5.2 : Improved DLF
---------------------

**2018-04-25**

- DLF improvements:

  - Digital linear filter (DLF) method for the Fourier transform can now be
    carried out without spline, providing 0 for ``pts_per_dec`` (or any
    integer smaller than 1).

  - Combine kernel from ``fht`` and ``ffht`` into ``dlf``, hence separate DLF
    from other calculations, as is done with QWE (``qwe`` for ``hqwe`` and
    ``fqwe``).

  - Bug fix regarding ``transform.get_spline_values``; a DLF with
    ``pts_per_dec`` can now be shorter then the corresponding filter.


v1.5.1 : Improved docs
----------------------

**2018-02-24**

- Documentation:

  - Simplifications: avoid duplication as much as possible between the website
    (empymod.github.io), the manual
    (`empymod.readthedocs.io <https://empymod.readthedocs.io>`_), and the
    ``README`` (github.com/empymod/empymod).

    - Website has now only *Features* and *Installation* in full, all other
      information comes in the form of links.
    - ``README`` has only information in the form of links.
    - Manual contains the ``README``, and is basically the main document for
      all information.

  - Improvements: Change some remaining ``md``-syntax to ``rst``-syntax.

  - FHT -> DLF: replace FHT as much as possible, without breaking backwards
    compatibility.


v1.5.0 : Hankel filter wer_201_2018
-----------------------------------

**2018-01-02**

- Minimum parameter values can now be set and verified with
  ``utils.set_minimum`` and ``utils.get_minimum``.

- New Hankel filter ``wer_201_2018``.

- ``opt=parallel`` has no effect if ``numexpr`` is not built against Intel's
  VML. (Use ``import numexpr; numexpr.use_vml`` to see if your ``numexpr`` uses
  VML.)

- Bug fixes

- Version of manuscript submission to geophysics for the DLF article.


v1.4.x
""""""


v1.4.4 : TE/TM split
--------------------

**2017-09-18**

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


v1.4.2 : Final submission version of Geophysics paper
-----------------------------------------------------

**2017-06-04**

- Bugfix: Fixed squeeze in ``model.analytical`` with ``solution='dsplit'``.

- Version of final submission of manuscript to Geophysics.


v1.4.1 : Own organisation github.com/empymod
--------------------------------------------

**2017-05-30**

[This was meant to be 1.4.0, but due to a setup/pypi/anaconda-issue I had to
push it to 1.4.1; so there isn't really a version 1.4.0.]

- New home: empymod.github.io as entry point, and the project page on
  github.com/empymod. All empymod-repos moved to the new home.

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


v1.3.x
"""""""


v1.3.0 : New transforms QUAD (Hankel) and FFT (Fourier)
-------------------------------------------------------

**2017-03-30**

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


v1.2.x
""""""


v1.2.1 : Installable via pip and conda
--------------------------------------

**2017-03-11**

- Change default filter from ``key_401_2009`` to ``key_201_2009`` (because of
  warning regarding 401 pt filter in source code of ``DIPOLE1D``.)

- Since 06/02/2017 installable via pip/conda.

- Bug fixes


v1.2.0 : Bipole
---------------

**2017-02-02**

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


v1.1.x
""""""


v1.1.0 : Include source bipole
------------------------------

**2016-12-22**

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


v1.0.x
""""""


v1.0.0 : Initial release
------------------------

**2016-11-29**

- Initial release; state of manuscript submission to geophysics.
