# Changelog

## latest

* [17/01/2017]
    - Cleaned-up wavenumber-domain routine.
    - Initiated tests (Status 0/52).
    - theta/phi -> azimuth/dip; easier to understand

* [14/01/2017]
    - General modelling routine `bipole` (replaces `srcbipole`): Model the
      EM field for arbitrarily oriented, finite length bipole sources and
      receivers.
    - Loads of changes in `utils`, with regards to the new routine `bipole` and
      with regards to verbosity. Moved all warnings out from `transform` and
      `model` into `utils`.
    - Add kernel count (printed if verb > 0).

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

## v1.0.0 - *2016-11-29*

* Initial release; state of manuscript submission to geophysics.
