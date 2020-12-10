Getting started
###############


Installation
------------

You can install empymod either via **conda**:

.. code-block:: console

   conda install -c conda-forge empymod

or via **pip**:

.. code-block:: console

   pip install empymod

Required are Python version 3.6 or higher and the modules `SciPy` and `Numba`.

The modeller empymod comes with add-ons (``empymod.scripts``). These add-ons
provide some very specific, additional functionalities. Some of these add-ons
have additional, optional dependencies such as matplotlib. See the
*Add-ons*-section for their documentation.

If you are new to Python I recommend using a Python distribution, which will
ensure that all dependencies are met, specifically properly compiled versions
of NumPy and SciPy; I recommend using `Anaconda
<https://www.anaconda.com/distribution>`_. If you install Anaconda you can
simply start the *Anaconda Navigator*, add the channel **conda-forge** and
**empymod** will appear in the package list and can be installed with a click.

Using NumPy and SciPy with the Intel Math Kernel Library (*mkl*) can
significantly improve computation time. You can check if ``mkl`` is used via
``conda list``: The entries for the BLAS and LAPACK libraries should contain
something with ``mkl``, not with ``openblas``. To enforce it you might have to
create a file ``pinned``, containing the line ``libblas[build=*mkl]`` in the
folder ``path-to-your-conda-env/conda-meta/``.

The structure of empymod is:

- **model.py**: EM modelling; principal end-user facing routines.
- **utils.py**: Utilities such as checking input parameters.
- **kernel.py**: Kernel of empymod, computes the wavenumber-domain
  electromagnetic response. Plus analytical, frequency-domain full- and
  half-space solutions.
- **transform.py**: Methods to carry out the required Hankel transform from
  wavenumber to space domain and Fourier transform from frequency to time
  domain.
- **filters.py**: Filters for the *Digital Linear Filters* method DLF (Hankel
  and Fourier transforms).

.. note::

    Until v2 empymod did not use Numba but instead optionally NumExpr. Use
    **v1.10.x** if you cannot use Numba or want to use NumExpr. However, no
    new features will land in v1, only bugfixes.


Usage
-----

The main modelling routine is :func:`empymod.model.bipole`, which can compute
the electromagnetic frequency- or time-domain field due to arbitrary finite
electric or magnetic bipole sources, measured by arbitrary finite electric or
magnetic bipole receivers. The model is defined by horizontal resistivity and
anisotropy, horizontal and vertical electric permittivities and horizontal and
vertical magnetic permeabilities. By default, the electromagnetic response is
normalized to source and receiver of 1 m length, and source strength of 1 A.

A simple frequency-domain example, with most of the parameters left at the
default value:

.. code-block:: python

    >>> import empymod
    >>> import numpy as np
    >>> # x-directed bipole source: x0, x1, y0, y1, z0, z1
    >>> src = [-50, 50, 0, 0, -100, -100]
    >>> # x-directed dipole source-array: x, y, z, azimuth, dip
    >>> rec = [np.arange(1, 11)*500, np.zeros(10), -200, 0, 0]
    >>> # layer boundaries
    >>> depth = [0, -300, -1000, -1050]
    >>> # layer resistivities
    >>> res = [1e20, .3, 1, 50, 1]
    >>> # Frequency
    >>> freq = 1
    >>> # Compute the electric field due to an electric source at 1 Hz.
    >>> # [msrc = mrec = True (default)]
    >>> EMfield = empymod.bipole(src, rec, depth, res, freq, verb=4)
    ~
    :: empymod START  ::  v2.0.0
    ~
       depth       [m] :  -1050, -1000, -300, 0
       res     [Ohm.m] :  1 50 1 0.3 1E+20
       aniso       [-] :  1 1 1 1 1
       epermH      [-] :  1 1 1 1 1
       epermV      [-] :  1 1 1 1 1
       mpermH      [-] :  1 1 1 1 1
       mpermV      [-] :  1 1 1 1 1
       direct field    :  Comp. in wavenumber domain
       frequency  [Hz] :  1
       Hankel          :  DLF (Fast Hankel Transform)
         > Filter      :  Key 201 (2009)
         > DLF type    :  Standard
       Loop over       :  None (all vectorized)
       Source(s)       :  1 bipole(s)
         > intpts      :  1 (as dipole)
         > length  [m] :  100
         > strength[A] :  0
         > x_c     [m] :  0
         > y_c     [m] :  0
         > z_c     [m] :  -100
         > azimuth [°] :  0
         > dip     [°] :  0
       Receiver(s)     :  10 dipole(s)
         > x       [m] :  500 - 5000 : 10  [min-max; #]
                       :  500 1000 1500 2000 2500 3000 3500 4000 4500 5000
         > y       [m] :  0 - 0 : 10  [min-max; #]
                       :  0 0 0 0 0 0 0 0 0 0
         > z       [m] :  -200
         > azimuth [°] :  0
         > dip     [°] :  0
       Required ab's   :  11
    ~
    :: empymod END; runtime = 0:00:00.005536 :: 1 kernel call(s)
    ~
    >>> print(EMfield)
    [  1.68809346e-10 -3.08303130e-10j  -8.77189179e-12 -3.76920235e-11j
      -3.46654704e-12 -4.87133683e-12j  -3.60159726e-13 -1.12434417e-12j
       1.87807271e-13 -6.21669759e-13j   1.97200208e-13 -4.38210489e-13j
       1.44134842e-13 -3.17505260e-13j   9.92770406e-14 -2.33950871e-13j
       6.75287598e-14 -1.74922886e-13j   4.62724887e-14 -1.32266600e-13j]


A good starting point is the :ref:`sphx_glr_examples`-gallery or [Wert17b]_,
and more detailed information can be found in [Wert17]_. The description of all
parameters can be found in the API documentation for
:func:`empymod.model.bipole`.


Coordinate system
-----------------

The used coordinate system is either a

- Left-Handed System (LHS), where Easting is the :math:`x`-direction, Northing
  the :math:`y`-direction, and positive :math:`z` is pointing downwards;
- Right-Handed System (RHS), where Easting is the :math:`x`-direction, Northing
  the :math:`y`-direction, and positive :math:`z` is pointing upwards.

Have a look at the example :ref:`sphx_glr_examples_coordinate_system.py` for
further explanations.


Theory
------

The code is principally based on

- [HuTS15]_ for the wavenumber-domain computation (``kernel``),
- [Key12]_ for the DLF and QWE transforms,
- [SlHM10]_ for the analytical half-space solutions, and
- [Hami00]_ for the FFTLog.

See these publications and all the others given in the :doc:`references`, if
you are interested in the theory on which empymod is based. Another good
reference is [ZiSl19]_. The book derives in great detail the equations for
layered-Earth CSEM modelling.


Contributing
------------

New contributions, bug reports, or any kind of feedback is always welcomed!
Have a look at the `Projects <https://github.com/emsig/empymod/projects>`_ on
GitHub to get an idea of things that could be implemented. The best way for
interaction is at https://github.com/emsig. If you prefer to contact me
outside of GitHub use the contact form on my personal website,
https://werthmuller.org.

To install empymod from source, you can download the latest version from GitHub
and install it in your python distribution via:

.. code-block:: console

   python setup.py install

Please make sure your code follows the pep8-guidelines by using, for instance,
the python module ``flake8``, and also that your code is covered with
appropriate tests. Just get in touch if you have any doubts.


Tests and benchmarks
--------------------

The modeller comes with a test suite using ``pytest``. If you want to run the
tests, just install ``pytest`` and run it within the ``empymod``-top-directory.

.. code-block:: console

    > pip install pytest coveralls pytest-flake8 pytest-mpl
    > # and then
    > cd to/the/empymod/folder  # Ensure you are in the right directory,
    > ls -d */                  # your output should look the same.
    docs/  empymod/  examples/  tests/
    > # pytest will find the tests, which are located in the tests-folder.
    > # simply run
    > pytest --cov=empymod --flake8 --mpl

It should run all tests successfully. Please let me know if not!

Note that installations of ``empymod`` via conda or pip do not have the
test-suite included. To run the test-suite you must download ``empymod`` from
GitHub.

There is also a benchmark suite using *airspeed velocity*, located in the
`emsig/empymod-asv <https://github.com/emsig/empymod-asv>`_-repository. The
results of my machine can be found in the `emsig/empymod-bench
<https://github.com/emsig/empymod-bench>`_, its rendered version at
`emsig.github.io/empymod-asv <https://emsig.github.io/empymod-asv/>`_.


License
-------

Copyright 2016-2020 The empymod Developers.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

See the LICENSE- and NOTICE-files on GitHub for more information.
