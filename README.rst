empymod
=======

|docs| |tests| |coverage|

Electromagnetic modeller in Python
----------------------------------

The electromagnetic modeller `empymod` can model electric or magnetic responses
due to a three-dimensional electric or magnetic source in a layered-earth model
with vertical transverse isotropy, electric permittivity, and magnetic
permeability, from very low frequencies to very high frequencies.

Installation & requirements
---------------------------

Just add the path to `empymod` to your python-path variable.

Alternatively, to install it in your python distribution (linux), run:

.. code:: bash

   python setup.py install

Required are python version 3.4 or higher and the modules `NumPy` and `SciPy`.
If you want to run parts of the kernel in parallel, the module `numexpr` is
required additionally.


Documentation
-------------

The manual for `empymod` can be found at http://empymod.readthedocs.io. Consult
the manual for further information and a list of TODOs.


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


Notice
------

This product includes software that was initially (till 01/2017) developed at
*The Mexican Institute of Petroleum IMP* (*Instituto Mexicano del Petróleo*,
http://www.imp.mx). The project was funded through *The Mexican National
Council of Science and Technology* (*Consejo Nacional de Ciencia y Tecnología*,
http://www.conacyt.mx).


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


Citation
--------

I am in the process of publishing an article regarding `empymod`, and I will
put the info here once it is a reality. If you publish results for which you
used `empymod`, please consider citing this article. Also consider citing the
two articles given above, *Hunziker et al, 2015*, and *Key, 2012*, without
which `empymod` would not exist.


.. |docs| image:: https://readthedocs.org/projects/empymod/badge/?version=latest
    :alt: Docs Status
    :target: http://empymod.readthedocs.io/en/latest/?badge=latest

.. |tests| image:: https://travis-ci.org/prisae/empymod.png?branch=master
    :alt: Test Status
    :target: https://travis-ci.org/prisae/empymod/

.. |coverage| image:: https://coveralls.io/repos/github/prisae/empymod/badge.svg?branch=master
    :alt: Coverage
    :target: https://coveralls.io/github/prisae/empymod?branch=master
