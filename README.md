# empymod

The electromagnetic modeller **empymod** can model electric or magnetic
responses due to a three-dimensional electric or magnetic source in a
layered-earth model with vertical transverse isotropic (VTI) resistivity, VTI
electric permittivity, and VTI magnetic permeability, from very low frequencies
(DC) to very high frequencies (GPR). The calculation is carried out in the
wavenumber-frequency domain, and various Hankel- and Fourier-transform methods
are included to transform the responses into the space-frequency and space-time
domains.


## More information

For information regarding installation, requirements, documentation, examples,
contributing and so on: [empymod.github.io](https://empymod.github.io).


## Installation from source

You can clone or download the latest version from GitHub and either add the
path to `empymod` to your python-path variable, or install it in your python
distribution via:

```bash
python setup.py install
```

## Testing

The modeller comes with a test suite using `pytest`. If you want to run the
tests, just install `pytest` and run it within the `empymod`-top-directory.

```bash
pytest
```

Pytest will find the tests, which are located in the tests-folder. It should
run all tests successfully. Please let me know if not!

Note that installations of `empymod` via conda or pip do not have the
test-suite included. To run the test-suite you must download `empymod` from
GitHub.


## Roadmap

A collection of ideas of what could be added or improved in empymod. Please
get in touch if you would like to tackle one of these problems!

- Additional modelling routines:
    - **Ramp waveform**
    - **Arbitrary waveform**
    - **TEM (in-loop, coincident loop, ...):** the problems to tackle are
      mainly zero-offset, loop integration, and windowing.
    - Improve the GPR-routine
    - Load and save functions to store and load model information
      (resistivity model, acquisition parameters, and modelling parameters)
      together with the modelling data.

- Additional analytical functions (semi-analytical); if possible
    - Complete fullspace (el./mag. src/rec); space-time domain
    - Extend diffusive halfspace solution to magnetic sources and receivers;
      space-frequency and space-time domains
    - Complete halfspace

- **Extend `fQWE` and `fftlog` to use cosine:** At the moment, `fqwe` and
  `fftlog` are implemented with the sine-transform. It would not be too much
  work to make them flexible to handle sine- and cosine-transforms. Having this
  flexibility we could calculate the step-off response with the cosine versions
  of `fftlog`, `fqwe`, and `ffht` instead of subtracting the step-on from the
  DC value. (Check how it works with `fft`!)

- Module to design digital filters
    - **Hankel transform (almost ready)**
    - Extend to Fourier transform

- Inversion: Inversion routines, preferably a selection of different ones.

- Improve documentation
    - Add actual equations, instead of only references to them.

- Extend examples (example-notebooks):
    - Add different methods (e.g. DC)
    - Reproduce published results

- A `cython`, `numba`, or pure C/C++ implementation of the `kernel` and the
  `transform` modules. Maybe not worth it, as it may improve speed, but
  decrease accessibility. Both at the same time would be nice. A fast
  C/C++-version for calculations (inversions), and a Python-version to
  tinker with for interested folks. (Probably combined with default
  parallelisation, removing the `numexpr` variant.)
- Abstraction of the code.

- GUI.

- Add a benchmark suite, e.g. http://asv.readthedocs.io, in addition to the
  testing suite.

- Add some clever checks, e.g. as in Key (2012): abort loops if the field
  is strongly attenuated.

- Move empymod from channel 'prisae' to 'conda-forge'.


## Citation

I am in the process of publishing an article in Geophysics regarding empymod,
and I will put the info here once it is a reality. If you publish results for
which you used empymod, please consider citing this article. Meanwhile, you
could cite the Geophysical Tutorial:

> Werthmüller, D., 2017, Getting started with controlled-source electromagnetic
> 1D modeling: The Leading Edge, 36, 352-355; DOI:
> [10.1190/tle36040352.1](http://dx.doi.org/10.1190/tle36040352.1).

Also consider citing the two articles given below, Hunziker et al. (2015)
and Key (2012), without which empymod would not exist:

> Hunziker, J., J. Thorbecke, and E. Slob, 2015, The electromagnetic response in
> a layered vertical transverse isotropic medium: A new look at an old problem:
> Geophysics, 80, F1-F18; DOI:
> [10.1190/geo2013-0411.1](http://dx.doi.org/10.1190/geo2013-0411.1).

> Key, K., 2012, Is the fast Hankel transform faster than quadrature?:
> Geophysics, 77, F21-F30; DOI:
> [10.1190/GEO2011-0237.1](http://dx.doi.org/10.1190/GEO2011-0237.1).

All releases have additionally a Zenodo-DOI, provided on the 
[release-page](https://github.com/empymod/empymod/releases).


Notice
------

This product includes software that was initially (till 01/2017) developed at
*The Mexican Institute of Petroleum IMP*
([Instituto Mexicano del Petróleo](http://www.gob.mx/imp)). The project was
funded through *The Mexican National Council of Science and Technology*
([Consejo Nacional de Ciencia y Tecnología](http://www.conacyt.mx)).

License
-------

Copyright 2016-2017 Dieter Werthmüller

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

[http://www.apache.org/licenses/LICENSE-2.0](http://www.apache.org/licenses/LICENSE-2.0)

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

See the *LICENSE*-file in the root directory for a full reprint of the Apache
License.
