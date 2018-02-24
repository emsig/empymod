Roadmap
#######

A collection of ideas of what could be added or improved in empymod. Please get
in touch if you would like to tackle one of these problems!

- **Additional modelling routines**

  - ``tdem`` (**TEM**) [`empymod#8
    <https://github.com/empymod/empymod/issues/8>`_]: Issues that have to be
    addressed: ramp waveform, windowing, loop integration, zero-offset
    (coincident loop).

    - in-loop
    - coincident loop
    - ...

  - **Ramp waveform** [`empymod#7
    <https://github.com/empymod/empymod/issues/7>`_]
  - **Arbitrary waveform** [`empymod#7
    <https://github.com/empymod/empymod/issues/7>`_]
  - Improve the GPR-routine [`empymod#9
    <https://github.com/empymod/empymod/issues/9>`_]
  - Load and save functions to easily store and load model information
    (resistivity model, acquisition parameters, and modelling parameters)
    together with the modelling data (using ``pickle`` or ``shelve``).


- **Inversion** [`empyscripts#1
  <https://github.com/empymod/empyscripts/issues/1>`_]: Inversion routines,
  preferably a selection of different ones.


- Additional (semi-)analytical functions (where possible)

  - Complete full-space (electric and magnetic source and receiver); space-time
    domain
  - Extend diffusive half-space solution to magnetic sources and receivers;
    space-frequency and space-time domains
  - Complete half-space


- Fourier transform

  - Change ``fft`` to use discrete sine/cosine transforms instead, as all other
    Fourier transforms
  - If previous step is successful, clean up the internal decisions
    (``utils.check_time``) when to use sine/cosine transform (not consistent at
    the moment, some choice only exists with ``ffht`` impulse responses,
    ``fqwe`` and ``fftlog`` use sine for impulse, and all three use sine for
    step-on responses and cosine for step-off responses)


- Hankel transform

  - Add the ``fht``-module from FFTLog for the Hankel transform.


- Extend examples (example-notebooks)

  - Add different methods (e.g. DC)
  - Reproduce published results


- A ``cython``, ``numba``, or pure C/C++ implementation of the ``kernel`` and
  the ``transform`` modules. Maybe not worth it, as it may improve speed, but
  decrease accessibility. Both at the same time would be nice. A fast
  C/C++-version for calculations (inversions), and a Python-version to tinker
  with for interested folks. (Probably combined with default parallelisation,
  removing the ``numexpr`` variant.)

- Abstraction of the code.

- GUI.

- Add a benchmark suite, e.g. http://asv.readthedocs.io, in addition to the
  testing suite.

- Add some clever checks, e.g. as in Key (2012): abort loops if the field is
  strongly attenuated (more relevant if once an inversion is implemented).

- Move empymod from channel 'prisae' to 'conda-forge' (pros/cons?).
