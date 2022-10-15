Maintenance
===========

Status reports and other tools to have the checks all in one place, for quick
QC.

Quick overview / QC
-------------------

All possible badges of the CI chain. Definitely check this before making a
release.

- .. image:: https://github.com/emsig/empymod/actions/workflows/linux.yml/badge.svg
     :target: https://github.com/emsig/empymod/actions/workflows/linux.yml
     :alt: GitHub Actions linux
  .. image:: https://github.com/emsig/empymod/actions/workflows/macos_windows.yml/badge.svg
     :target: https://github.com/emsig/empymod/actions/workflows/macos_windows.yml
     :alt: GitHub Actions macos & windows
  .. image:: https://github.com/emsig/empymod/actions/workflows/linkcheck.yml/badge.svg
     :target: https://github.com/emsig/empymod/actions/workflows/linkcheck.yml
     :alt: GitHub Actions linkcheck
  .. image:: https://readthedocs.org/projects/empymod/badge/?version=latest
     :target: https://empymod.emsig.xyz/en/latest
     :alt: Documentation Status

  Ensure CI and docs are passing.

- .. image:: https://img.shields.io/pypi/v/empymod.svg
     :target: https://pypi.python.org/pypi/empymod
     :alt: PyPI
  .. image:: https://img.shields.io/conda/v/conda-forge/empymod.svg
     :target: https://anaconda.org/conda-forge/empymod
     :alt: conda-forge

  Ensure latest version is deployed on PyPI and conda.

- .. image:: https://coveralls.io/repos/github/emsig/empymod/badge.svg?branch=main
     :target: https://coveralls.io/github/emsig/empymod?branch=main
     :alt: Coveralls
  .. image:: https://app.codacy.com/project/badge/Grade/0412e617e8cd42fea05303fe490b09b5
     :target: https://www.codacy.com/gh/emsig/empymod/dashboard?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=emsig/empymod&amp;utm_campaign=Badge_Grade
     :alt: Codacy

  Check CI coverage and code quality is good.

- .. image:: https://img.shields.io/badge/benchmark-asv-blue.svg?style=flat
     :target: https://emsig.xyz/empymod-asv
     :alt: Airspeed Velocity

  Check Benchmarks are run up to the latest version.

- .. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.593094.svg
     :target: https://doi.org/10.5281/zenodo.593094
     :alt: Zenodo DOI

  Check Zenodo is linking to the latest release.


Info from ReadTheDocs
---------------------

To check the environment in which the documentation was built.

.. ipython::

    In [1]: import empymod
       ...: empymod.Report(
       ...:     ['sphinx', 'numpydoc', 'sphinx_design', 'sphinx_numfig',
       ...:      'sphinx_gallery', 'memory_profiler', 'pydata_sphinx_theme',
       ...:      'sphinx_automodapi', 'ipykernel', ]
       ...: )
