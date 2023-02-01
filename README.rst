.. image:: https://raw.github.com/emsig/logos/main/empymod/empymod-logo.png
   :target: https://emsig.xyz
   :alt: empymod logo

|

.. image:: https://img.shields.io/pypi/v/empymod-plain.svg
   :target: https://pypi.python.org/pypi/empymod-plain/
   :alt: PyPI
.. image:: https://img.shields.io/badge/python-3.7+-blue.svg
   :target: https://www.python.org/downloads/
   :alt: Supported Python Versions
.. image:: https://img.shields.io/badge/platform-linux,win,osx-blue.svg
   :target: https://anaconda.org/conda-forge/empymod/
   :alt: Linux, Windows, OSX

|


The branch `plain <https://github.com/emsig/empymod/tree/plain>`_ is a
stripped-down version of ``empymod``, where ``numba`` is removed and the
computation is done in plain python. The only dependency is SciPy. Its
execution is slow, as the kernel is not compiled. It is mainly meant to be used
for ``pyodido`` and similar, browser-based client-side applications, which need
plain python wheels. The name of the PyPI-package is ``empymod-plain``, but the
module itself is ``empymod``.

The version number follows the ``empymod`` version number, with an added
``.post1`` to distinguish it.

**Only use this version if you are sure why. In general it is always advisable
to use the regular empymod package.**

Open-source full 3D electromagnetic modeller for 1D VTI media

- **Website:** https://emsig.xyz
- **Documentation:** https://empymod.emsig.xyz
- **Source Code:** https://github.com/emsig/empymod
- **Bug reports:** https://github.com/emsig/empymod/issues
- **Contributing:** https://empymod.emsig.xyz/en/latest/dev
- **Contact:** see https://emsig.xyz
- **Zenodo:** https://doi.org/10.5281/zenodo.593094


Available through ``pip`` (not ``conda``), meant primarily for ``micropip``
for, e.g., ``jupyterlite``/``pyodide``.
