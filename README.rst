#######
empymod
#######

.. image:: https://readthedocs.org/projects/empymod/badge/?version=latest
   :target: http://empymod.readthedocs.io/en/latest/?badge=latest
   :alt: Documentation Status
.. image:: https://travis-ci.org/empymod/empymod.svg?branch=master
   :target: https://travis-ci.org/empymod/empymod
   :alt: Travis-CI
.. image:: https://coveralls.io/repos/github/empymod/empymod/badge.svg?branch=master
   :target: https://coveralls.io/github/empymod/empymod?branch=master
   :alt: Coveralls

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

For more information (regarding installation, usage, add-ons, contributing,
roadmap, bug reports, etc) see

- **Website**: `empymod.github.io <https://empymod.github.io>`_,
- **Documentation**: `empymod.rtfd.io <https://empymod.rtfd.io>`_,
- **Source Code**:
  `github.com/empymod/empymod <https://github.com/empymod/empymod>`_,
- **Add-ons**:
  `github.com/empymod/empyscripts <https://github.com/empymod/empyscripts>`_,
- **Examples**:
  `github.com/empymod/example-notebooks
  <https://github.com/empymod/example-notebooks>`_.


Citation
========

If you publish results for which you used empymod, please give credit by citing
[Werthmuller_2017]_:

    Werthmüller, D., 2017, An open-source full 3D electromagnetic modeler for
    1D VTI media in Python: empymod: Geophysics, 82(6), WB9--WB19; DOI:
    `10.1190/geo2016-0626.1 <http://doi.org/10.1190/geo2016-0626.1>`_.


All releases have a Zenodo-DOI, provided on the `release-page
<https://github.com/empymod/empymod/releases>`_. Also consider citing
[Hunziker_et_al_2015]_ and [Key_2012]_, without which empymod would not exist.


License information
===================

Copyright 2016-2018 Dieter Werthmüller

Licensed under the Apache License, Version 2.0. See the ``LICENSE``- and
``NOTICE``-files on GitHub or the documentation for more information.
