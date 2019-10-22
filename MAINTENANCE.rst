Maintainers Guide
=================

A release is currently done completely manually, no automatic deployment is
set up.


Making a release
----------------

1. Update:

   - ``CHANGELOG``
   - ``setup.py``: Version number, remove '.dev?'.
   - ``empymod/__init__.py``: Check version number, remove '.dev?'.

2. Check syntax of README::

       python setup.py --long-description | rst2html.py --no-raw > index.html

3. Remove any old stuff (just in case)::

       rm -rf build/ dist/ empymod.egg-info/

4. Push it to GitHub, create a release tagging it

5. Get the Zenodo-DOI and add it to release notes

6. Ensure ``python3-setuptools`` is installed::

       sudo apt install python3-setuptools

7. Create tar and wheel::

       python setup.py sdist
       python setup.py bdist_wheel

8. Test it on testpypi (requires ~/.pypirc)::

       ~/anaconda3/bin/twine upload dist/* -r testpypi

   Optionally test it already in conda if skeleton builds::

       conda skeleton pypi --pypi-url https://test.pypi.io/pypi/ empymod

9. Push it to PyPi (requires ~/.pypircs)::

       ~/anaconda3/bin/twine upload dist/*

10. conda build

    Has to be done outside of ~/, because conda skeleton cannot handle, at the
    moment, the encrypted home
    (https://conda.io/docs/build_tutorials/pkgs.html).


    1. Install miniconda in /opt::

           wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh;
           bash miniconda.sh -b -p /opt/miniconda/miniconda
           export PATH="/opt/miniconda/miniconda/bin:$PATH"
           conda update conda
           conda install -y conda-build anaconda-client
           conda config --set anaconda_upload yes
           anaconda login

    2. Now to the conda-build part::

           conda skeleton pypi empymod
           conda build --python 3.5 empymod
           conda build --python 3.6 empymod
           conda build --python 3.7 empymod

    3. Convert for all platforms::

           conda convert --platform all /opt/miniconda/miniconda/conda-bld/linux-64/empymod-[version]-py35_0.tar.bz2
           conda convert --platform all /opt/miniconda/miniconda/conda-bld/linux-64/empymod-[version]-py36_0.tar.bz2
           conda convert --platform all /opt/miniconda/miniconda/conda-bld/linux-64/empymod-[version]-py37_0.tar.bz2

    4. Upload them::

           anaconda upload osx-64/*
           anaconda upload win-*/*
           anaconda upload linux-32/*

    5. Logout::

           anaconda logout

10. Post-commit changes

    - ``setup.py``: Bump number, add '.dev0' to version number
    - ``empymod/__init__.py``: Bump number, add '.dev0' to version number
