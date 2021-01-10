
.. image:: https://raw.githubusercontent.com/emsig/empymod-logo/master/logo-empymod-plain-250px.png
   :target: https://emsig.github.io
   :alt: empymod logo
   
----

.. image:: https://readthedocs.org/projects/empymod/badge/?version=latest
   :target: https://empymod.readthedocs.io/en/latest
   :alt: Documentation Status
.. image:: https://github.com/emsig/empymod/workflows/pytest/badge.svg?branch=master
   :target: https://github.com/emsig/empymod/actions
   :alt: GitHub Actions
.. image:: https://coveralls.io/repos/github/emsig/empymod/badge.svg?branch=master
   :target: https://coveralls.io/github/emsig/empymod?branch=master
   :alt: Coveralls
.. image:: https://app.codacy.com/project/badge/Grade/f2db5d9f0f3a4cd6b8457a6e1c47203a
   :target: https://www.codacy.com/gh/emsig/empymod/dashboard?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=emsig/empymod&amp;utm_campaign=Badge_Grade
   :alt: Codacy
.. image:: https://img.shields.io/badge/benchmark-asv-blue.svg?style=flat
   :target: https://emsig.github.io/empymod-asv/
   :alt: Airspeed Velocity
.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.593094.svg
   :target: https://doi.org/10.5281/zenodo.593094
   :alt: Zenodo DOI

.. sphinx-inclusion-marker

The electromagnetic modeller **empymod** can model electric or magnetic
responses due to a three-dimensional electric or magnetic source in a
layered-earth model with vertical transverse isotropic (VTI) resistivity, VTI
electric permittivity, and VTI magnetic permeability, from very low frequencies
(DC) to very high frequencies (GPR). The computation is carried out in the
wavenumber-frequency domain, and various Hankel- and Fourier-transform methods
are included to transform the responses into the space-frequency and space-time
domains.


More information
================

For more information regarding installation, usage, contributing, bug reports,
and much more, see

- **Website**: https://emsig.github.io
- **Documentation**: https://empymod.readthedocs.io
- **Source Code**: https://github.com/emsig/empymod
- **Examples**: https://empymod.readthedocs.io/en/stable/examples

Features
========

- Computes the complete (diffusion and wave phenomena) 3D electromagnetic field
  in a layered-earth model including vertical transverse isotropic (VTI)
  resistivity, VTI electric permittivity, and VTI magnetic permeability, for
  electric and magnetic sources as well as electric and magnetic receivers.

- Modelling routines:

  - ``bipole``: arbitrary oriented, finite length bipoles with given source
    strength; space-frequency and space-time domains.
  - ``dipole``: infinitesimal small dipoles oriented along the principal axes,
    normalized field; space-frequency and space-time domains.
  - ``loop``: arbitrary oriented loop source measured by arbitrary oriented,
    finite length electric or magnetic dipole or loop receivers;
    space-frequency and space-time domains.
  - ``dipole_k``: as ``dipole``, but returns the wavenumber-frequency domain
    response.
  - ``gpr``: computes the ground-penetrating radar response for given central
    frequency, using a Ricker wavelet (experimental).
  - ``analytical``: interface to the analytical, space-frequency and space-time
    domain solutions.

- Hankel transforms (wavenumber-frequency to space-frequency transform):

  - DLF: Digital Linear Filters (using included filters or providing own ones)
  - QWE: Quadrature with extrapolation
  - QUAD: Adaptive quadrature

- Fourier transforms (space-frequency to space-time transform):

  - DLF: Digital Linear Filters (using included filters or providing own ones)
  - QWE: Quadrature with extrapolation
  - FFTLog: Logarithmic Fast Fourier Transform
  - FFT: Fast Fourier Transform

- Analytical, space-frequency and space-time domain solutions:

  - Complete full-space (electric and magnetic sources and receivers);
    space-frequency domain
  - Diffusive half-space (electric sources and receivers); space-frequency and
    space-time domains:

    - Direct wave (= diffusive full-space solution)
    - Reflected wave
    - Airwave (semi-analytical in the case of step responses)

- Add-ons (``empymod.scripts``):

  The add-ons for empymod provide some very specific, additional
  functionalities:

  - ``tmtemod``: Return up- and down-going TM/TE-mode contributions for
    x-directed electric sources and receivers, which are located in the same
    layer.
  - ``fdesign``: Design digital linear filters for the Hankel and Fourier
    transforms.

- Hidden features (incomplete list, see manual for more):

  - Models with frequency-dependent resistivity (e.g., Cole-Cole IP).
  - Space-Laplace domain computation for the numerical and analytical
    solutions.


Installation
============

You can install empymod either via **conda**:

.. code-block:: console

   conda install -c conda-forge empymod

or via **pip**:

.. code-block:: console

   pip install empymod

Required are Python version 3.6 or higher and the modules `SciPy` and `Numba`.
Consult the installation notes in the `manual
<https://empymod.readthedocs.io/en/stable/manual.html#installation>`_ for more
information regarding installation and requirements.


Citation
========

If you publish results for which you used empymod, please give credit by citing
`Werthmüller (2017)  <https://doi.org/10.1190/geo2016-0626.1>`_:

    Werthmüller, D., 2017, An open-source full 3D electromagnetic modeler for
    1D VTI media in Python: empymod: Geophysics, 82(6), WB9-WB19; DOI:
    `10.1190/geo2016-0626.1 <https://doi.org/10.1190/geo2016-0626.1>`_.

All releases have a Zenodo-DOI, which can be found on `10.5281/zenodo.593094
<https://doi.org/10.5281/zenodo.593094>`_. Also consider citing
`Hunziker et al. (2015) <https://doi.org/10.1190/geo2013-0411.1>`_ and
`Key (2012) <https://doi.org/10.1190/geo2011-0237.1>`_, without which
empymod would not exist.


License information
===================

Copyright 2016-2021 The empymod Developers.

Licensed under the Apache License, Version 2.0. See the LICENSE- and
NOTICE-files or the documentation for more information.
