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
import warnings

import libdlf
import numpy as np

__all__ = ['DigitalFilter', 'Hankel', 'Fourier']


def __dir__():
    return __all__


FILTERS = {
    'hankel': dict.fromkeys(libdlf.hankel.__all__),
    'fourier': dict.fromkeys(libdlf.fourier.__all__)
}


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
        self.factor = np.around([self.base[1]/self.base[0]], 15)


class Hankel:

    def __init__(self):
        for k, v in FILTERS['hankel'].items():
            setattr(self, k, v)

    def __getattribute__(self, name):
        if name in FILTERS['hankel'].keys():
            if FILTERS['hankel'][name] is None:
                FILTERS['hankel'][name] = load_filter(name, 'hankel.'+name)
            return FILTERS['hankel'][name]
        else:
            return object.__getattribute__(self, name)


class Fourier:

    def __init__(self):
        for k, v in FILTERS['fourier'].items():
            setattr(self, k, v)

    def __getattribute__(self, name):
        if name in FILTERS['fourier'].keys():
            if FILTERS['fourier'][name] is None:
                FILTERS['fourier'][name] = load_filter(name, 'fourier.'+name)
            return FILTERS['fourier'][name]
        else:
            return object.__getattribute__(self, name)


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


# DEPRECATIONS - REMOVE in v3.0 #

def _deprecate_filter(func):

    def newfn():
        name = func.__name__
        if 'CosSin' in name:
            new = name.replace('_CosSin', '')
            ftype = 'Fourier'
        else:
            ftype = 'Hankel'
            new = name

        if 'kong_61' in name:
            new += 'b'

        msg = (
            f"Calling `empymod.filters.{name}()` is deprecated and will be "
            f"removed in v3.0; use `empymod.filters.{ftype}().{new}`."
        )
        warnings.warn(msg, FutureWarning)
        return func()

    return newfn


# 1. Hankel DLF

@_deprecate_filter
def kong_61_2007():
    """Deprecated; just for backwards compatibility until v3.0."""
    return Hankel().kong_61_2007b


@_deprecate_filter
def kong_241_2007():
    """Deprecated; just for backwards compatibility until v3.0."""
    return Hankel().kong_241_2007


@_deprecate_filter
def key_101_2009():
    """Deprecated; just for backwards compatibility until v3.0."""
    return Hankel().key_101_2009


@_deprecate_filter
def key_201_2009():
    """Deprecated; just for backwards compatibility until v3.0."""
    return Hankel().key_201_2009


@_deprecate_filter
def key_401_2009():
    """Deprecated; just for backwards compatibility until v3.0."""
    return Hankel().key_401_2009


@_deprecate_filter
def anderson_801_1982():
    """Deprecated; just for backwards compatibility until v3.0."""
    return Hankel().anderson_801_1982


@_deprecate_filter
def key_51_2012():
    """Deprecated; just for backwards compatibility until v3.0."""
    return Hankel().key_51_2012


@_deprecate_filter
def key_101_2012():
    """Deprecated; just for backwards compatibility until v3.0."""
    return Hankel().key_101_2012


@_deprecate_filter
def key_201_2012():
    """Deprecated; just for backwards compatibility until v3.0."""
    return Hankel().key_201_2012


@_deprecate_filter
def wer_201_2018():
    """Deprecated; just for backwards compatibility until v3.0."""
    return Hankel().wer_201_2018


# 2. Fourier DLF (cosine/sine)


@_deprecate_filter
def key_81_CosSin_2009():
    """Deprecated; just for backwards compatibility until v3.0."""
    return Fourier().key_81_2009


@_deprecate_filter
def key_241_CosSin_2009():
    """Deprecated; just for backwards compatibility until v3.0."""
    return Fourier().key_241_2009


@_deprecate_filter
def key_601_CosSin_2009():
    """Deprecated; just for backwards compatibility until v3.0."""
    return Fourier().key_601_2009


@_deprecate_filter
def key_101_CosSin_2012():
    """Deprecated; just for backwards compatibility until v3.0."""
    return Fourier().key_101_2012


@_deprecate_filter
def key_201_CosSin_2012():
    """Deprecated; just for backwards compatibility until v3.0."""
    return Fourier().key_201_2012
