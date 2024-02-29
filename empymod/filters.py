r"""
Filters for the *Digital Linear Filter* (DLF) method for the Hankel
[Ghos70]_) and the Fourier ([Ande75]_) transforms.

To calculate the `dlf.factor` I used

.. code-block:: python

    np.around([dlf.base[1]/dlf.base[0]], 15)

The filters `kong_61_2007` and `kong_241_2007` from [Kong07]_, and
`key_101_2009`, `key_201_2009`, `key_401_2009`, `key_81_CosSin_2009`,
`key_241_CosSin_2009`, and `key_601_CosSin_2009` from [Key09]_ are taken from
*DIPOLE1D*, [Key09]_, which can be downloaded at
https://marineemlab.ucsd.edu/Projects/Occam/1DCSEM ([1DCSEM]_). *DIPOLE1D* is
distributed under the license GNU GPL version 3 or later. Kerry Key gave his
written permission to re-distribute the filters under the Apache License,
Version 2.0 (email from Kerry Key to Dieter Werthmüller, 21 November 2016).

The filters `anderson_801_1982` from [Ande82]_ and `key_51_2012`,
`key_101_2012`, `key_201_2012`, `key_101_CosSin_2012`, and
`key_201_CosSin_2012`, all from [Key12]_, are taken from the software
distributed with [Key12]_ and available at https://software.seg.org/2012/0003
([SEG-2012-003]_). These filters are distributed under the SEG license.

The filter `wer_201_2018` was designed with the add-on `fdesign`, see
https://github.com/emsig/article-fdesign.

"""
# Copyright 2016 The emsig community.
#
# This file is part of empymod.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License.  You may obtain a copy
# of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  See the
# License for the specific language governing permissions and limitations under
# the License.


import os
import libdlf
import numpy as np

__all__ = ['DigitalFilter', 'Hankel', 'Fourier']


def __dir__():
    return __all__


# 0. Filter Class and saving/loading routines

class DigitalFilter:
    r"""Simple Class for Digital Linear Filters.


    Parameters
    ----------
    name : str
        Name of the DFL.

    savename = str
        Name with which the filter is saved. If None (default) it is set to the
        same value as `name`.

    filter_coeff = list of str
        By default, the following filter coefficients are checked:

            ``filter_coeff = ['j0', 'j1', 'sin', 'cos']``

        This accounts for the standard Hankel and Fourier DLF in CSEM
        modelling. However, additional coefficient names can be provided via
        this parameter (in list format).

    """

    def __init__(self, name, savename=None, filter_coeff=None):
        r"""Add filter name."""
        self.name = name
        if savename is None:
            self.savename = name
        else:
            self.savename = savename

        # Define coefficient names.
        self.filter_coeff = ['j0', 'j1', 'sin', 'cos']
        if filter_coeff is not None:  # Additional, user provided.
            self.filter_coeff.extend(filter_coeff)

    def tofile(self, path='filters'):
        r"""Save filter values to ASCII-files.

        Store the filter base and the filter coefficients in separate files
        in the directory `path`; `path` can be a relative or absolute path.

        Examples
        --------
        >>> import empymod
        >>> # Load a filter
        >>> filt = empymod.filters.wer_201_2018()
        >>> # Save it to pure ASCII-files
        >>> filt.tofile()
        >>> # This will save the following three files:
        >>> #    ./filters/wer_201_2018_base.txt
        >>> #    ./filters/wer_201_2018_j0.txt
        >>> #    ./filters/wer_201_2018_j1.txt

        """

        # Get name of filter
        name = self.savename

        # Get absolute path, create if it doesn't exist
        path = os.path.abspath(path)
        os.makedirs(path, exist_ok=True)

        # Save filter base
        basefile = os.path.join(path, name + '_base.txt')
        with open(basefile, 'w') as f:
            self.base.tofile(f, sep="\n")

        # Save filter coefficients
        for val in self.filter_coeff:
            if hasattr(self, val):
                attrfile = os.path.join(path, name + '_' + val + '.txt')
                with open(attrfile, 'w') as f:
                    getattr(self, val).tofile(f, sep="\n")

    def fromfile(self, path='filters'):
        r"""Load filter values from ASCII-files.

        Load filter base and filter coefficients from ASCII files in the
        directory `path`; `path` can be a relative or absolute path.

        Examples
        --------
        >>> import empymod
        >>> # Create an empty filter;
        >>> # Name has to be the base of the text files
        >>> filt = empymod.filters.DigitalFilter('my-filter')
        >>> # Load the ASCII-files
        >>> filt.fromfile()
        >>> # This will load the following three files:
        >>> #    ./filters/my-filter_base.txt
        >>> #    ./filters/my-filter_j0.txt
        >>> #    ./filters/my-filter_j1.txt
        >>> # and store them in filt.base, filt.j0, and filt.j1.

        """

        # Get name of filter
        name = self.savename

        # Get absolute path
        path = os.path.abspath(path)

        # Get filter base
        basefile = os.path.join(path, name + '_base.txt')
        with open(basefile, 'r') as f:
            self.base = np.fromfile(f, sep="\n")

        # Get filter coefficients
        for val in self.filter_coeff:
            attrfile = os.path.join(path, name + '_' + val + '.txt')
            if os.path.isfile(attrfile):
                with open(attrfile, 'r') as f:
                    setattr(self, val, np.fromfile(f, sep="\n"))

        # Add factor
        self.factor = np.around([self.base[1]/self.base[r]], 15)


HANKEL = {k: None for k in libdlf.hankel.__all__}
FOURIER = {k: None for k in libdlf.fourier.__all__}


class Hankel:
    def __getattr__(self, name):
        if name in HANKEL.keys():
            if HANKEL[name] is None:
                HANKEL[name] = load_filter(name, 'hankel.'+name)
            return HANKEL[name]
        else:
            raise AttributeError(
                f"'{self.__class__.__name__}' object has no attribute '{name}'"
            )


class Fourier:
    def __getattr__(self, name):
        if name in FOURIER.keys():
            if FOURIER[name] is None:
                FOURIER[name] = load_filter(name, 'fourier.'+name)
            return FOURIER[name]
        else:
            raise AttributeError(
                f"'{self.__class__.__name__}' object has no attribute '{name}'"
            )


# TODO:
# - DEPRECATION WARNINGS
# - NEEDS TO WORK FOR FILTER INSTANCE, STRING, AND LIBDLF INSTANCE

def load_filter(name, libdlfname):
    ftype, fname = libdlfname.split('.')

    TransformClass = [Hankel, Fourier][ftype == 'hankel']

    if not hasattr(TransformClass, fname):
        dd = getattr(getattr(libdlf, ftype), fname)
        filter_coeff = dd.values
        values = dd()

        dlf = DigitalFilter(name, filter_coeff=filter_coeff)
        dlf.base = values[0]
        for i, val in enumerate(filter_coeff):
            setattr(dlf, val, values[i+1])
        dlf.factor = np.around([dlf.base[1]/dlf.base[0]], 15)

        setattr(TransformClass, libdlfname, dlf)

    return getattr(TransformClass, libdlfname)


# 1. Hankel DLF

def kong_61_2007():
    r"""Kong 61 pt Hankel filter, as published in [Kong07]_.

    Taken from file `FilterModules.f90` provided with [1DCSEM]_.

    License: `Apache License, Version 2.0,
    <https://www.apache.org/licenses/LICENSE-2.0>`_.

    """
    return Hankel().kong_61_2007b


def kong_241_2007():
    r"""Kong 241 pt Hankel filter, as published in [Kong07]_.

    Taken from file `FilterModules.f90` provided with [1DCSEM]_.

    License: `Apache License, Version 2.0,
    <https://www.apache.org/licenses/LICENSE-2.0>`_.

    """
    return Hankel().kong_241_2007


def key_101_2009():
    r"""Key 101 pt Hankel filter, as published in [Key09]_.

    Taken from file `FilterModules.f90` provided with [1DCSEM]_.

    License: `Apache License, Version 2.0,
    <https://www.apache.org/licenses/LICENSE-2.0>`_.

    """
    return Hankel().key_101_2009


def key_201_2009():
    r"""Key 201 pt Hankel filter, as published in [Key09]_.

    Taken from file `FilterModules.f90` provided with [1DCSEM]_.

    License: `Apache License, Version 2.0,
    <https://www.apache.org/licenses/LICENSE-2.0>`_.

    """
    return Hankel().key_201_2009


def key_401_2009():
    r"""Key 401 pt Hankel filter, as published in [Key09]_.

    Taken from file `FilterModules.f90` provided with [1DCSEM]_.

    License: `Apache License, Version 2.0,
    <https://www.apache.org/licenses/LICENSE-2.0>`_.

    """
    return Hankel().key_401_2009


def anderson_801_1982():
    r"""Anderson 801 pt Hankel filter, as published in [Ande82]_.

    Taken from file `wa801Hankel.txt` provided with [SEG-2012-003]_.

    License: https://software.seg.org/disclaimer.txt.

    """
    return Hankel().anderson_801_1982


def key_51_2012():
    r"""Key 51 pt Hankel filter, as published in [Key12]_.

    Taken from file `kk51Hankel.txt` provided with [SEG-2012-003]_.

    License: https://software.seg.org/disclaimer.txt.

    """
    return Hankel().key_51_2012


def key_101_2012():
    r"""Key 101 pt Hankel filter, as published in [Key12]_.

    Taken from file `kk101Hankel.txt` provided with [SEG-2012-003]_.

    License: https://software.seg.org/disclaimer.txt.

    """
    return Hankel().key_101_2012


def key_201_2012():
    r"""Key 201 pt Hankel filter, as published in [Key12]_.

    Taken from file `kk201Hankel.txt` provided with [SEG-2012-003]_.

    License: https://software.seg.org/disclaimer.txt.

    """
    return Hankel().key_201_2012


def wer_201_2018():
    r"""Werthmüller 201 pt Hankel filter, 2018.

    Designed with the empymod add-on `fdesign`, see
    https://github.com/emsig/article-fdesign.

    License: `Apache License, Version 2.0,
    <https://www.apache.org/licenses/LICENSE-2.0>`_.

    """
    return Hankel().wer_201_2018


# 2. Fourier DLF (cosine/sine)


def key_81_CosSin_2009():
    r"""Key 81 pt CosSin filter, as published in [Key09]_.

    Taken from file `FilterModules.f90` provided with [1DCSEM]_.

    License: `Apache License, Version 2.0,
    <https://www.apache.org/licenses/LICENSE-2.0>`_.

    """
    return Fourier().key_81_2009


def key_241_CosSin_2009():
    r"""Key 241 pt CosSin filter, as published in [Key09]_.

    Taken from file `FilterModules.f90` provided with [1DCSEM]_.

    License: `Apache License, Version 2.0,
    <https://www.apache.org/licenses/LICENSE-2.0>`_.

    """
    return Fourier().key_241_2009


def key_601_CosSin_2009():
    r"""Key 601 pt CosSin filter, as published in [Key09]_.

    Taken from file `FilterModules.f90` provided with [1DCSEM]_.

    License: `Apache License, Version 2.0,
    <https://www.apache.org/licenses/LICENSE-2.0>`_.

    """
    return Fourier().key_601_2009


def key_101_CosSin_2012():
    r"""Key 101 pt CosSin filter, as published in [Key12]_.

    Taken from file `kk101CosSin.txt` provided with [SEG-2012-003]_.

    License: https://software.seg.org/disclaimer.txt.

    """
    return Fourier().key_101_2012


def key_201_CosSin_2012():
    r"""Key 201 pt CosSin filter, as published in [Key12]_.

    Taken from file `kk201CosSin.txt` provided with [SEG-2012-003]_.

    License: https://software.seg.org/disclaimer.txt.

    """
    return Fourier().key_201_2012
