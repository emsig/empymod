Maintainers Guide
=================

Releases of ``empymod`` are currently done manually. This is the 'recipe'.


Making a release
----------------

1. Update:

   - ``CHANGELOG``
   - ``empymod/__init__.py``: Update version number.

2. Remove any old stuff (just in case)::

       rm -rf build/ dist/ empymod.egg-info/

3. Push it to GitHub, create a release tagging it
   (ensure correct tag is in local home with ``python setup.py --version``).

4. Get the Zenodo-DOI and add it to release notes; also RTFD, which might have
   to be triggered first.

5. Create tar and wheel::

       python setup.py sdist
       python setup.py bdist_wheel

6. Push it to PyPi (requires ~/.pypircs)::

       ~/anaconda3/bin/twine upload dist/*

7. ``conda`` build:

   Has to be done outside of ~/, because conda skeleton cannot handle, at the
   moment, the encrypted home
   (https://conda.io/docs/build_tutorials/pkgs.html). Also, ensure you leave
   any current conda environment with ``conda deactivate``.


   1. Install miniconda in /opt::

          wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh;
          bash miniconda.sh -b -p /opt/miniconda/miniconda
          export PATH="/opt/miniconda/miniconda/bin:$PATH"
          conda update conda
          conda install -y conda-build anaconda-client
          conda config --set anaconda_upload yes
          anaconda login

   2. Create the skeleton for conda from PyPi::

          conda skeleton pypi empymod

      Edit the ``empymod/meta.yml`` in the following way:

      - under ``build:``, add ``preserve_egg_dir: True``
      - under ``requirement: host:``, add ``- setuptools_scm``

   3. Now to the conda-build part::

          conda build --python 3.5 empymod
          conda build --python 3.6 empymod
          conda build --python 3.7 empymod

   4. Convert for all platforms::

          conda convert --platform all /opt/miniconda/miniconda/conda-bld/linux-64/empymod-[version]-py35_0.tar.bz2
          conda convert --platform all /opt/miniconda/miniconda/conda-bld/linux-64/empymod-[version]-py36_0.tar.bz2
          conda convert --platform all /opt/miniconda/miniconda/conda-bld/linux-64/empymod-[version]-py37_0.tar.bz2

   5. Upload them::

          anaconda upload osx-64/*
          anaconda upload win-*/*
          anaconda upload linux-32/*

   6. Logout::

          anaconda logout


Useful things
-------------

- If unsure, test it first on testpypi (requires ~/.pypirc)::

       ~/anaconda3/bin/twine upload dist/* -r testpypi

- If unsure, test the test-pypi for conda if the skeleton builds::

       conda skeleton pypi --pypi-url https://test.pypi.io/pypi/ empymod

- If there were changes to README, check it with::

       python setup.py --long-description | rst2html.py --no-raw > index.html

- If it fails, you might have to install ``python3-setuptools``::

       sudo apt install python3-setuptools
