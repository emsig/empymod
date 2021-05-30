About
=====

The electromagnetic modeller **empymod** can model electric or magnetic
responses due to a three-dimensional electric or magnetic source in a
layered-earth model with vertical transverse isotropic (VTI) resistivity, VTI
electric permittivity, and VTI magnetic permeability, from very low frequencies
(DC) to very high frequencies (GPR). The computation is carried out in the
wavenumber-frequency domain, and various Hankel- and Fourier-transform methods
are included to transform the responses into the space-frequency and space-time
domains.


Features
--------

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


Related ecosystem
-----------------

empymod has been successfully used as a forward modeller in `pyGIMLi
<https://pygimli.org>`_, and could potentially also be used with `SimPEG
<https://simpeg.xyz>`_. Get in touch if you are interested in these
developments.

See also the note about the `EM & Potential Geo-Exploration Python Ecosystem
<https://emsig.xyz/#related-ecosystem>`_ on `emsig.xyz <https://emsig.xyz>`_.
