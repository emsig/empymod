# Changelog

## v1.4.4 - *2017-09-18*
* Bug-fix unrelated to the code itself (setup/pypi/anaconda-issue).
  Don't do that anymore!

## v1.4.3 - *2017-09-18*

* Add TE/TM split to diffusive ee-halfspace solution.

* Improve `kernel.wavenumber` for fullspaces.

* Extended `fQWE` and `fftlog` to be able to use the cosine-transform. Now the
  cosine-transform with the real-part frequency response is used internally if
  a switch-off response (`signal=-1`) is required, rather than calculating the
  switch-on response (with sine-transform and imaginary-part frequency
  response) and subtracting it from the DC value.

* Bug fixes

* Version of CSEM book.

## v1.4.2 - *2017-06-04*

* Bugfix: Fixed squeeze in `model.analytical` with `solution='dsplit'`.

* Version of final submission of manuscript to Geophysics.


## v1.4.1 - *2017-05-30*
* Bug-fix unrelated to the code itself (setup/pypi/anaconda-issue).


## v1.4.0 - *2017-05-30*

* New home: [empymod.github.io](https://empymod.github.io) as entry point,
  and the project page on [github.com/empymod](https://github.com/empymod).
  All empymod-repos moved to the new home:
    * /prisae/empymod -> /empymod/empymod
    * /prisae/empymod-notebooks -> /empymod/example-notebooks
    * /prisae/empymod-geo2017 -> /empymod/article-geo2017
    * /prisae/empymod-tle2017 -> /empymod/article-tle2017

* Modelling routines:
    * New modelling routine `model.analytical`, which serves as a front-end to
      `kernel.fullspace` or `kernel.halfspace`.
    * Remove legacy routines `model.time` and `model.frequency`.  They are
      covered perfectly by `model.dipole`.
    * Improved switch-off response (calculate and subtract from DC).
    * `xdirect` adjustments:
        * `isfullspace` now respects `xdirect`.
        * Removed `xdirect` from `model.wavenumber` (set to `False`).

* Kernel:
    * Modify `kernel.halfspace` to use same input as other kernel functions.
    * Include time-domain ee halfspace solution into `kernel.halfspace`;
      possible to obtain direct, reflected, and airwave separately, as well as
      only fullspace solution (all for the diffusive approximation).


## v1.3.0 - *2017-03-30*

* Add additional transforms and improve QWE:
    * Conventional adaptive quadrature (QUADPACK) for the Hankel transform;
    * Conventional FFT for the Fourier transform.
    * Add `diff_quad` to `htarg`/`ftarg` of QWE, a switch parameter for
      QWE/QUAD.
    * Change QWE/QUAD switch from comparing first interval to comparing all
      intervals.
    * Add parameters for QUAD (a, b, limit) into `htarg`/`ftarg` for QWE.

* Allow `htarg`/`ftarg` as dict additionally to list/tuple.

* Improve `model.gpr`.

* Internal changes:
    * Rename internally the sine/cosine filter from `fft` to `ffht`, because of
      the addition of the Fast Fourier Transform `fft`.

* Clean-up repository
    * Move `notebooks` to /prisae/empymod-notebooks
    * Move `publications/Geophysics2017` to /prisae/empymod-geo2017
    * Move `publications/TheLeadingEdge2017` to /prisae/empymod-tle2017

* Bug fixes and documentation improvements


## v1.2.1 - *2017-03-11*

* Change default filter from `key_401_2009` to `key_201_2009` (because of
  warning regarding 401 pt filter in source code of `DIPOLE1D`.)

* Since 06/02/2017 installable via pip/conda.

* Bug fixes


## v1.2.0 - *2017-02-02*

* New routine:
    * General modelling routine `bipole` (replaces `srcbipole`): Model the
      EM field for arbitrarily oriented, finite length bipole sources and
      receivers.

* Added a test suite:
    * Unit-tests of small functions.
    * Framework-tests of the bigger functions:
        * Comparing to status quo (regression tests),
        * Comparing to known analytical solutions,
        * Comparing different options to each other,
        * Comparing to other 1D modellers (EMmod, DIPOLE1D, GREEN3D).
    * Incorporated with Travis CI and Coveralls.

* Internal changes:
    * Add kernel count (printed if verb > 1).
    * `numexpr` is now only required if `opt=='parallel'`. If `numexpr` is not
      found, `opt` is reset to `None` and a warning is printed.
    * Cleaned-up wavenumber-domain routine.
    * theta/phi -> azimuth/dip; easier to understand.
    * Refined verbosity levels.
    * Lots of changes in `utils`, with regards to the new routine `bipole` and
      with regards to verbosity. Moved all warnings out from `transform` and
      `model` into `utils`.

* Bug fixes


## v1.1.0 - *2016-12-22*

* New routines:
    * New `srcbipole` modelling routine: Model an arbitrarily oriented, finite
      length bipole source.
    * Merge `frequency` and `time` into `dipole`. (`frequency` and `time` are
      still available.)
    * `dipole` now supports multiple sources.

* Internal changes:
    * Replace `get_Gauss_Weights` with `scipy.special.p_roots`
    * `jv(0,x)`, `jv(1,x)` -> `j0(x)`, `j1(x)`
    * Replace `param_shape` in `utils` with `_check_var` and `_check_shape`.
    * Replace `xco` and `yco` by `angle` in `kernel.fullspace`
    * Replace `fftlog` with python version.
    * Additional sine-/cosine-filters: `key_81_CosSin_2009`,
      `key_241_CosSin_2009`, and `key_601_CosSin_2009`.

* Bug fixes


## v1.0.0 - *2016-11-29*

* Initial release; state of manuscript submission to geophysics.
