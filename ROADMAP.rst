Roadmap
#######

A collection of ideas of what could be added or improved in empymod. Please get
in touch if you would like to tackle one of these problems!

- **Additional modelling routines and utilities**

  - **TEM**
    [`empymod#8 <https://github.com/empymod/empymod/issues/8>`_]:
    Windowing functionality and more possibilities for loop measurements:

    - concentric loops
    - coincident loops
    - arbitrary shaped loops

  - **Ramp waveform**
    [`empymod#7 <https://github.com/empymod/empymod/issues/7>`_]
  - **Arbitrary waveform**
    [`empymod#7 <https://github.com/empymod/empymod/issues/7>`_]
  - Improve the GPR-routine
    [`empymod#9 <https://github.com/empymod/empymod/issues/9>`_]
  - Load and save functions to easily store and load model information
    (resistivity model, acquisition parameters, and modelling parameters)
    together with the modelling data (using ``pickle`` or ``shelve``).
    Probably easier after implementation of the abstraction
    [`empymod#14 <https://github.com/empymod/empymod/issues/14>`_].


- **Inversion** [`empymod#20 <https://github.com/empymod/empymod/issues/20>`_]:
  Inversion routines, preferably a selection of different ones.

  - Add some clever checks, e.g. as in Key (2012): abort loops if the field
    is strongly attenuated.


- Additional (semi-)analytical functions (where possible)

  - Complete full-space (electric and magnetic source and receiver); space-time
    domain
  - Extend diffusive half-space solution to magnetic sources and receivers;
    space-frequency and space-time domains
  - Complete half-space


- Transforms

  - Fourier

    - Change ``fft`` to use discrete sine/cosine transforms instead, as all
      other Fourier transforms
    - If previous step is successful, clean up the internal decisions
      (``utils.check_time``) when to use sine/cosine transform (not consistent
      at the moment, some choice only exists with ``ffht`` impulse responses,
      ``fqwe`` and ``fftlog`` use sine for impulse, and all three use sine for
      step-on responses and cosine for step-off responses)


  - Hankel

    - Add the ``fht``-module from FFTLog for the Hankel transform.


- Extend examples (example-notebooks)

  - Add different methods (e.g. DC)
  - Reproduce published results


- GUI.

- Move empymod from channel 'prisae' to 'conda-forge' (pros/cons?).
