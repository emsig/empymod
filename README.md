# empymod

[![readthedocs](https://readthedocs.org/projects/empymod/badge/?version=latest)](https://empymod.readthedocs.io/en/latest/?badge=latest)
[![travis-ci](https://travis-ci.org/empymod/empymod.png?branch=master)](https://travis-ci.org/empymod/empymod/)
[![coveralls](https://coveralls.io/repos/github/empymod/empymod/badge.svg?branch=master)](https://coveralls.io/github/empymod/empymod?branch=master)


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

For the add-ons to empymod see
[github.com/empymod/empyscripts](https://github.com/empymod/empyscripts).


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

See `ROADMAP.rst`.


## Citation

If you publish results for which you used empymod, please give credit by citing
this article:

> Werthmüller, D., 2017, An open-source full 3D electromagnetic modeler for 1D
> VTI media in Python: empymod: Geophysics, 82(6), WB9-WB19; DOI:
> [10.1190/geo2016-0626.1](http://doi.org/10.1190/geo2016-0626.1).

Also consider citing Hunziker et al. (2015) and Key (2012), without which
empymod would not exist:

> Hunziker, J., J. Thorbecke, and E. Slob, 2015, The electromagnetic response in
> a layered vertical transverse isotropic medium: A new look at an old problem:
> Geophysics, 80(1), F1-F18; DOI:
> [10.1190/geo2013-0411.1](http://doi.org/10.1190/geo2013-0411.1).
>  
> Key, K., 2012, Is the fast Hankel transform faster than quadrature?:
> Geophysics, 77(3), F21-F30; DOI:
> [10.1190/geo2011-0237.1](http://doi.org/10.1190/geo2011-0237.1).

All releases have a Zenodo-DOI, provided on the
[release-page](https://github.com/empymod/empymod/releases).


## Notice

This software was initially (till 01/2017) developed with funding from
*The Mexican National Council of Science and Technology*
([Consejo Nacional de Ciencia y Tecnología](http://www.conacyt.gob.mx)),
carried out at *The Mexican Institute of Petroleum IMP*
([Instituto Mexicano del Petróleo](http://www.gob.mx/imp)).


## License

Copyright 2016-2018 Dieter Werthmüller

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
