Getting started
###############

The main modelling routine is :func:`empymod.model.bipole`, which can compute
the electromagnetic frequency- or time-domain field due to arbitrary finite
electric or magnetic dipole sources, measured by arbitrary finite electric or
magnetic dipole receivers. The model is defined by horizontal resistivity and
anisotropy, horizontal and vertical electric permittivities and horizontal and
vertical magnetic permeabilities. By default, the electromagnetic response is
normalized to source and receiver of 1 m length, and source strength of 1 A.


Basic example
-------------

A simple frequency-domain example, where we keep most of the parameters left at
the default value:

.. ipython::

  In [1]: import empymod
     ...: import numpy as np
     ...: import matplotlib.pyplot as plt

First we define the survey parameters: source and receiver locations, and
source frequencies.

.. ipython::

  In [1]: # x-directed dipole source: x0, x1, y0, y1, z0, z1
     ...: source = [-50, 50, 0, 0, -100, -100]
     ...: # Source frequency
     ...: frequency = 1
     ...: # Receiver offsets
     ...: offsets = np.arange(1, 11)*500
     ...: # x-directed dipole receiver-array: x, y, z, azimuth, dip
     ...: receivers = [offsets, offsets*0, -200, 0, 0]

Next, we define the resistivity model:

.. ipython::

  In [1]: # Layer boundaries
     ...: depth = [0, -300, -1000, -1050]
     ...: # Layer resistivities
     ...: resistivities = [1e20, 0.3, 1, 50, 1]

And finally we compute the electromagnetic response at receiver locations:

.. ipython::

  In [1]: efield = empymod.bipole(
     ...:         src=source,
     ...:         rec=receivers,
     ...:         depth=depth,
     ...:         res=resistivities,
     ...:         freqtime=frequency,
     ...:         verb=4,
     ...: )

Let's plot the resulting responses:

.. ipython::

  @savefig basic_example.png width=4in
  In [1]: fig, ax = plt.subplots()
     ...: ax.semilogy(offsets/1e3, abs(efield.real), 'C0o-', label='|Real|')
     ...: ax.semilogy(offsets/1e3, abs(efield.imag), 'C1o-', label='|Imag|')
     ...: ax.legend()
     ...: ax.set_title('Electric field at receivers')
     ...: ax.set_xlabel('Offset (km)');
     ...: ax.set_ylabel('Electric field (V/m)');

A good starting point is the :ref:`empymod_gallery`-gallery or [Wert17b]_, and
more detailed information can be found in [Wert17]_. The description of all
parameters can be found in the API documentation for
:func:`empymod.model.bipole`.


Structure
---------

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


Coordinate system
-----------------

The used coordinate system is either a

- Left-Handed System (LHS), where Easting is the :math:`x`-direction, Northing
  the :math:`y`-direction, and positive :math:`z` is pointing downwards;
- Right-Handed System (RHS), where Easting is the :math:`x`-direction, Northing
  the :math:`y`-direction, and positive :math:`z` is pointing upwards.

Have a look at the example
:ref:`sphx_glr_gallery_educational_coordinate_system.py` for further
explanations.


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
