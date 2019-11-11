Maintainers Guide
=================


Making a release
----------------

1. Update ``CHANGELOG.rst``.

2. Push it to GitHub, create a release tagging it.

3. Tagging it on GitHub will automatically deploy it to PyPi, which in turn
   will create a PR for the conda-forge `feedstock
   <https://github.com/conda-forge/empymod-feedstock>`_. Merge that PR.

4. Release notes edits: (1) get and add the `Zenodo-DOI
   <https://doi.org/10.5281/zenodo.593094>`_; (b) add the readthedocs badge,
   you might have to trigger a build first.


Useful things
-------------

- If there were changes to README, check it with::

       python setup.py --long-description | rst2html.py --no-raw > index.html

- If unsure, test it first manually on testpypi (requires ~/.pypirc)::

       ~/anaconda3/bin/twine upload dist/* -r testpypi

- If unsure, test the test-pypi for conda if the skeleton builds::

       conda skeleton pypi --pypi-url https://test.pypi.io/pypi/ empymod

- If it fails, you might have to install ``python3-setuptools``::

       sudo apt install python3-setuptools


CI
--

- Testing on `Travis <https://travis-ci.org/empymod/empymod>`_, includes:

  - Tests using ``pytest``
  - Linting / code style with ``pytest-flake8``
  - Figures with ``pytest-mpl``
  - Ensure all http(s)-links work (``sphinx linkcheck``)

- Line-coverage with ``pytest-cov`` on `Coveralls
  <https://coveralls.io/github/empymod/empymod>`_
- Code-quality on `Codacy
  <https://app.codacy.com/manual/prisae/empymod/dashboard>`_
- Manual on `ReadTheDocs <https://empymod.readthedocs.io/en/latest>`_,
  including the Gallery (examples run each time).
- DOI minting on `Zenodo <https://doi.org/10.5281/zenodo.593094>`_
- Benchmarks with `Airspeed Velocity <https://empymod.github.io/empymod-asv>`_
  (``asv``) [currently manually]
- Automatically deploys if tagged:

  - `PyPi <https://pypi.org/project/empymod>`_
  - `conda -c conda-forge <https://anaconda.org/conda-forge/empymod>`_
