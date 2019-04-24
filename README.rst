
.. image:: https://raw.githubusercontent.com/empymod/logo/master/logo-empymod-plain-250px.png
   :target: https://empymod.github.io
   :alt: empymod logo
   
----

.. image:: https://readthedocs.org/projects/empymod/badge/?version=latest
   :target: http://empymod.readthedocs.io/en/latest
   :alt: Documentation Status
.. image:: https://travis-ci.org/empymod/empymod.svg?branch=master
   :target: https://travis-ci.org/empymod/empymod
   :alt: Travis-CI
.. image:: https://coveralls.io/repos/github/empymod/empymod/badge.svg?branch=master
   :target: https://coveralls.io/github/empymod/empymod?branch=master
   :alt: Coveralls
.. image:: https://img.shields.io/codacy/grade/b28ed3989ed248fe95e34288e43667b9/master.svg
   :target: https://www.codacy.com/app/prisae/empymod
   :alt: Codacy
.. image:: https://img.shields.io/badge/benchmark-asv-blue.svg?style=flat
   :target: https://empymod.github.io/empymod-asv
   :alt: Airspeed Velocity
.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.593094.svg
   :target: https://doi.org/10.5281/zenodo.593094
   :alt: Zenodo DOI

.. sphinx-inclusion-marker

The electromagnetic modeller **empymod** can model electric or magnetic
responses due to a three-dimensional electric or magnetic source in a
layered-earth model with vertical transverse isotropic (VTI) resistivity, VTI
electric permittivity, and VTI magnetic permeability, from very low frequencies
(DC) to very high frequencies (GPR). The calculation is carried out in the
wavenumber-frequency domain, and various Hankel- and Fourier-transform methods
are included to transform the responses into the space-frequency and space-time
domains.

See https://empymod.github.io/#features for a complete list of features.

More information
================

For more information regarding installation, usage, contributing, roadmap, bug
reports, and much more, see

- **Website**: https://empymod.github.io,
- **Documentation**: https://empymod.readthedocs.io,
- **Source Code**: https://github.com/empymod,
- **Examples**: https://github.com/empymod/empymod-examples.


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
