Steps to carry out for a new release
====================================

   1. Update CHANGELOG

   2. Update the version number in
      - `setup.py`
      - `docs/conf.py`
      - `empymod/__init__.py`

   3. Check and update everything in setup.py

   4. Remove any old stuff (just in case)

        rm -rf build/ dist/ empymod.egg-info/

   5. Push it to GitHub, create a release tagging it

   6. Create tar and wheel

        python setup.py sdist
        python setup.py bdist_wheel

   7. Test it on testpypi (requires ~/.pypirc)

        twine upload dist/* -r testpypi

   8. Push it to PyPi (requires ~/.pypirc)

        twine upload dist/*

   9. conda build

   Has to be done outside of ~/, because conda skeleton cannot handle, at the
   moment, the encrypted home.
   https://conda.io/docs/build_tutorials/pkgs.html


        # Install miniconda in /opt
        wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh;
        bash miniconda.sh -b -p /opt/miniconda/miniconda
        export PATH="/opt/miniconda/miniconda/bin:$PATH"
        conda update conda
        conda install conda-build anaconda-client
        conda config --set anaconda_upload yes
        anaconda login

        # Now to the conda-build part
        conda skeleton pypi empymod
        conda build --python 3.4 empymod
        conda build --python 3.5 empymod
        conda build --python 3.6 empymod

        # Convert for all platforms

        conda convert --platform all /opt/miniconda/miniconda/conda-bld/linux-64/empymod-1.2.0-py34_0.tar.bz2
        conda convert --platform all /opt/miniconda/miniconda/conda-bld/linux-64/empymod-1.2.0-py35_0.tar.bz2
        conda convert --platform all /opt/miniconda/miniconda/conda-bld/linux-64/empymod-1.2.0-py36_0.tar.bz2

        # Logout
        anaconda logout

   10. Update version number with a dev for future
      - `setup.py`
      - `docs/conf.py`
      - `empymod/__init__.py`
