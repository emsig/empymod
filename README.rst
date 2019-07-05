
.. image:: https://raw.githubusercontent.com/empymod/logo/master/logo-empymod-plain-250px.png
   :target: https://empymod.github.io
   :alt: empymod logo
   
----

.. sphinx-inclusion-marker

The electromagnetic modeller **empymod** can model electric or magnetic
responses due to a three-dimensional electric or magnetic source in a
layered-earth model with vertical transverse isotropic (VTI) resistivity, VTI
electric permittivity, and VTI magnetic permeability, from very low frequencies
(DC) to very high frequencies (GPR). The calculation is carried out in the
wavenumber-frequency domain, and various Hankel- and Fourier-transform methods
are included to transform the responses into the space-frequency and space-time
domains.


More information
================

For more information regarding installation, usage, contributing, roadmap, bug
reports, and much more, see

- **Website**: https://empymod.github.io,
- **Documentation**: https://empymod.readthedocs.io,
- **Source Code**: https://github.com/empymod,
- **Examples**: https://github.com/empymod/empymod-examples.


Features
========

- Calculates the complete (diffusion and wave phenomena) 3D electromagnetic
  field in a layered-earth model including vertical transverse isotropic (VTI)
  resistivity, VTI electric permittivity, and VTI magnetic permeability, for
  electric and magnetic sources as well as electric and magnetic receivers.

- Modelling routines:

  - ``bipole``: arbitrary oriented, finite length bipoles with given source
    strength; space-frequency and space-time domains.
  - ``dipole``: infinitesimal small dipoles oriented along the principal axes,
    normalized field; space-frequency and space-time domains.
  - ``wavenumber``: as ``dipole``, but returns the wavenumber-frequency domain
    response.
  - ``gpr``: calculates the ground-penetrating radar response for given central
    frequency, using a Ricker wavelet (experimental).
  - ``analytical``: interface to the analytical, space-frequency and space-time
    domain solutions.

- Hankel transforms (wavenumber-frequency to space-frequency transform):

  - Digital Linear Filters DLF (using included filters or providing own ones)
  - Quadrature with extrapolation QWE
  - Adaptive quadrature QUAD

- Fourier transforms (space-frequency to space-time transform):
  - Digital Linear Filters DLF (using included filters or providing own ones)
  - Quadrature with extrapolation QWE
  - Logarithmic Fast Fourier Transform FFTLog
  - Fast Fourier Transform FFT

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


Installation
============

You can install empymod either via ``conda``:

.. code-block:: console

   conda install -c prisae empymod

or via ``pip``:

.. code-block:: console

   pip install empymod

Required are Python version 3.5 or higher and the modules `NumPy` and `SciPy`.
Consult the installation notes in the `manual
<https://empymod.readthedocs.io/en/stable/manual.html#installation>`_ for more
information regarding installation and requirements.


Citation
========

If you publish results for which you used empymod, please give credit by citing
`Werthmüller (2017)  <http://doi.org/10.1190/geo2016-0626.1>`_:

    Werthmüller, D., 2017, An open-source full 3D electromagnetic modeler for
    1D VTI media in Python: empymod: Geophysics, 82(6), WB9--WB19; DOI:
    `10.1190/geo2016-0626.1 <http://doi.org/10.1190/geo2016-0626.1>`_.

All releases have a Zenodo-DOI, provided on the
`release-page <https://github.com/empymod/empymod/releases>`_.
Also consider citing
`Hunziker et al. (2015) <https://doi.org/10.1190/geo2013-0411.1>`_ and
`Key (2012) <https://doi.org/10.1190/geo2011-0237.1>`_, without which
empymod would not exist.


License information
===================

Copyright 2016-2019 Dieter Werthmüller

Licensed under the Apache License, Version 2.0. See the ``LICENSE``- and
``NOTICE``-files or the documentation for more information.
