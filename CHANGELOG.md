# Changelog

## v1.2.0 - *2017-02-02*

* New routine:
    - General modelling routine `bipole` (replaces `srcbipole`): Model the
      EM field for arbitrarily oriented, finite length bipole sources and
      receivers.

* Added a test suite:
    - Unit-tests of small functions.
    - Framework-tests of the bigger functions:
        - Comparing to status quo (regression tests),
        - Comparing to known analytical solutions,
        - Comparing different options to each other,
        - Comparing to other 1D modellers (EMmod, DIPOLE1D, GREEN3D).
    - Incorporated with Travis CI and Coveralls.

* Internal changes:
    - Add kernel count (printed if verb > 1).
    - `numexpr` is now only required if `opt=='parallel'`. If `numexpr` is not
      found, `opt` is reset to `None` and a warning is printed.
    - Cleaned-up wavenumber-domain routine.
    - theta/phi -> azimuth/dip; easier to understand.
    - Refined verbosity levels.
    - Lots of changes in `utils`, with regards to the new routine `bipole` and
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
