Making a release
================

Making a release is by now straight forward. Creating a «New Release» on GitHub
will trigger all the required things. However, there are still a few things to
do beforehand and a few things to check afterwards.

1. Update ``CHANGELOG.rst``.

2. Push it to GitHub, create a release tagging it.

3. Tagging it on GitHub will automatically deploy it to PyPi, which in turn
   will create a PR for the conda-forge `feedstock
   <https://github.com/conda-forge/empymod-feedstock>`_. The PR will be
   automatically merged. (Note: if the dependencies change or the minimum
   Python version or other installation things then the feedstock has to be
   updated manually!)

4. After releasing it, check that:

  - `PyPi <https://pypi.org/project/empymod>`_ deployed;
  - `conda-forge <https://anaconda.org/conda-forge/empymod>`_ deployed;
  - `Zenodo <https://doi.org/10.5281/zenodo.593094>`_ minted a DOI;
  - `empymod.emsig.xyz <https://empymod.emsig.xyz>`_ created a tagged version.


CI automatic and manual bits
----------------------------

Automatic
`````````

- Testing on `Github Actions <https://github.com/emsig/empymod/actions>`_
  includes:

  - Tests using ``pytest`` (Linux, MacOS, Windows)
  - Linting / code style with ``flake8``
  - Ensure all http(s)-links work (``sphinx -b linkcheck``)

- Line-coverage with ``pytest-cov`` on `Coveralls
  <https://coveralls.io/github/emsig/empymod>`_
- Code-quality on `Codacy
  <https://app.codacy.com/gh/emsig/empymod/dashboard>`_
- Manual on `ReadTheDocs <https://empymod.emsig.xyz/en/latest>`_
- DOI minting on `Zenodo <https://doi.org/10.5281/zenodo.593094>`_

Manual
``````

- Benchmarks with `Airspeed Velocity <https://emsig.xyz/empymod-asv>`_
  (``asv``)


Useful things
-------------

The following info was from the time when we still manually deployed. Now
every push to main triggers a test to Test-PyPI, so things can be verified
there. However, these hints my still be useful at some point.

- If there were changes to README, check it with::

       python setup.py --long-description | rst2html.py --no-raw > index.html

- If unsure about something, test it first on testpypi (requires ~/.pypirc)::

       ~/anaconda3/bin/twine upload dist/* -r testpypi

- If unsure, test the test-pypi for conda if the skeleton builds::

       conda skeleton pypi --pypi-url https://test.pypi.io/pypi/ empymod

- If it fails, you might have to install ``python3-setuptools``::

       sudo apt install python3-setuptools
