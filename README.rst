empymod
=======

Electromagnetic modeller in Python
----------------------------------

The electromagnetic modeller `empymod` can model electric or magnetic responses
due to a three-dimensional electric or magnetic source in a layered-earth model
with vertical transverse isotropy, electric permittivity, and magnetic
permeability, from very low frequencies to very high frequencies.

Installation & requirements
---------------------------

The easiest way to install the latest stable version of `empymod` is via
`conda`:

.. code:: bash

    conda install -c prisae empymod

or via `pip`:

.. code:: bash

   pip install empymod

Alternatively, you can download the latest version from GitHub and either add
the path to `empymod` to your python-path variable, or install it in your
python distribution via:

.. code:: bash

   python setup.py install

Required are python version 3.4 or higher and the modules `NumPy` and `SciPy`.
If you want to run parts of the kernel in parallel, the module `numexpr` is
required additionally.

If you are new to Python I recommend using a Python distribution, which will
ensure that all dependencies are met, specifically properly compiled versions
of `NumPy` and `SciPy`; I recommend using Anaconda (version 3.x;
`continuum.io/downloads <https://www.continuum.io/downloads>`_).  If you
install Anaconda you can simply start the *Anaconda Navigator*, add the channel
`prisae` and `empymod` will appear in the package list and can be installed
with a click.

Documentation
-------------

The manual of `empymod` can be found at `empymod.readthedocs.io
<http://empymod.readthedocs.io/en/stable>`_.


Citation
--------

I am in the process of publishing an article regarding `empymod`, and I will
put the info here once it is a reality. If you publish results for which you
used `empymod`, please consider citing this article. Also consider citing the
two articles given below, *Hunziker et al, 2015*, and *Key, 2012*, without
which `empymod` would not exist. (All releases have additionally a Zenodo-DOI,
provided on the `release-page <https://github.com/prisae/empymod/releases>`_.)


Notice
------

This product includes software that was initially (till 01/2017) developed at
*The Mexican Institute of Petroleum IMP* (*Instituto Mexicano del Petróleo*,
`gob.mx/imp <http://www.gob.mx/imp>`_). The project was funded through *The
Mexican National Council of Science and Technology* (*Consejo Nacional de
Ciencia y Tecnología*, `conacyt.mx <http://www.conacyt.mx>`_).


This product is a derivative work of the following two publications and their
publicly available software:

1. Hunziker, J., J. Thorbecke, and E. Slob, 2015, The electromagnetic response
   in a layered vertical transverse isotropic medium: A new look at an old
   problem: Geophysics, 80, F1-F18; DOI: `10.1190/geo2013-0411.1
   <http://dx.doi.org/10.1190/geo2013-0411.1>`_; Software:
   `software.seg.org/2015/0001 <http://software.seg.org/2015/0001>`_.

2. Key, K., 2012, Is the fast Hankel transform faster than quadrature?:
   Geophysics, 77, F21-F30; DOI: `10.1190/GEO2011-0237.1
   <http://dx.doi.org/10.1190/GEO2011-0237.1>`_; Software:
   `software.seg.org/2012/0003 <http://software.seg.org/2012/0003>`_.

Both pieces of software are published under the *SEG disclaimer*. Parts of the
modeller `emmod` from Hunziker et al, 2015, is furthermore released under the
*Common Public License Version 1.0 (CPL)*. See the *NOTICE*-file in the root
directory for more information and a reprint of the SEG disclaimer and the CPL.


License
-------

Copyright 2016-2017 Dieter Werthmüller

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

See the *LICENSE*-file in the root directory for a full reprint of the Apache
License.
