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

- **Additional modelling routines**
    - `tdem` (**TEM**)
      Issues that have to be adressed: ramp waveform, windowing, loop
      integration, zero-offset (coincident loop).
        - in-loop
        - coincident loop
        - ...
    - **Ramp waveform**
    - **Arbitrary waveform**
    - Improve the GPR-routine
    - Load and save functions to easily store and load model information
      (resistivity model, acquisition parameters, and modelling parameters)
      together with the modelling data (using `pickle` or `shelve`).


- **Inversion**: Inversion routines, preferably a selection of different ones.


- **Improve documentation**
    - Move main part from `empymod/__init__.py` to `docs/*.rst`
    - Add actual equations, instead of only references to them
    - Add general EM introduction and derivation
    - Add a few simple example of its usage including figures
    - Add more explanations regarding the different transforms


- Additional (semi-)analytical functions (where possible)
    - Complete full-space (electric and magnetic source and receiver);
      space-time domain
    - Extend diffusive half-space solution to magnetic sources and receivers;
      space-frequency and space-time domains
    - Complete half-space


- Fourier transform
    - Change `fft` to use discrete sine/cosine transforms instead, as all other
      Fourier transforms
    - If previous step is successful, clean up the internal decisions
      (`utils.check_time`) when to use sine/cosine transform (not consistent at
      the moment, some choice only exists with `ffht` impulse responses, `fqwe`
      and `fftlog` use sine for impulse, and all three use sine for step-on
      responses and cosine for step-off responses)


- Hankel transform
    - Add the `fht`-module from FFTLog for the Hankel transform.


- Module to design digital filters
    - **Hankel transform** (almost ready)
    - Extend to Fourier transform


- Extend examples (example-notebooks)
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
  is strongly attenuated (more relevant if once an inversion is implemented).

- Move empymod from channel 'prisae' to 'conda-forge' (pros/cons?).


## Citation

If you publish results for which you used empymod, please give credit by citing
this article:

> Werthmüller, D., 2017, An open-source full 3D electromagnetic modeler for 1D
> VTI media in Python: empymod: Geophysics, 82, WB9-WB19; DOI:
> [10.1190/geo2016-0626.1](http://doi.org/10.1190/geo2016-0626.1).

Also consider citing Hunziker et al. (2015) and Key (2012), without which
empymod would not exist:

> Hunziker, J., J. Thorbecke, and E. Slob, 2015, The electromagnetic response in
> a layered vertical transverse isotropic medium: A new look at an old problem:
> Geophysics, 80, F1-F18; DOI:
> [10.1190/geo2013-0411.1](http://doi.org/10.1190/geo2013-0411.1).
>  
> Key, K., 2012, Is the fast Hankel transform faster than quadrature?:
> Geophysics, 77, F21-F30; DOI:
> [10.1190/geo2011-0237.1](http://doi.org/10.1190/geo2011-0237.1).

All releases have a Zenodo-DOI, provided on the
[release-page](https://github.com/empymod/empymod/releases).


## Notice

This product includes software that was initially (till 01/2017) developed at
*The Mexican Institute of Petroleum IMP*
([Instituto Mexicano del Petróleo](http://www.gob.mx/imp)). The project was
funded through *The Mexican National Council of Science and Technology*
([Consejo Nacional de Ciencia y Tecnología](http://www.conacyt.mx)). Since
02/2017 it is a personal effort, and new contributors are welcome!


## License

Copyright 2016-2017 Dieter Werthmüller

Licensed under the Apache License, Version 2.0 (the "License"); you may not use
this file except in compliance with the License.  You may obtain a copy of the
License at

[http://www.apache.org/licenses/LICENSE-2.0](http://www.apache.org/licenses/LICENSE-2.0)

Unless required by applicable law or agreed to in writing, software distributed
under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
CONDITIONS OF ANY KIND, either express or implied.  See the License for the
specific language governing permissions and limitations under the License.

See the *LICENSE*-file in the root directory for a full reprint of the Apache
License.
