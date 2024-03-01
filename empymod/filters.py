r"""
Filters for the *Digital Linear Filter* (DLF) method for the Hankel
[Ghos70]_) and the Fourier ([Ande75]_) transforms.

Starting with v2.3.0, the actual filters are not stored here any longer, but
are loaded from **libdlf** (https://github.com/emsig/libdlf). Each filter is
documented in its own docstring, also indicating the license under which it is
distributed.

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


# Filter-cache
FILTERS = {
    'hankel': dict.fromkeys(libdlf.hankel.__all__),
    'fourier': dict.fromkeys(libdlf.fourier.__all__)
}


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
        >>> filt = empymod.filters.Hankel().wer_201_2018
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


class _BaseFilter:
    """Base class for wrappers loading filters from libdlf."""

    def __init__(self, ftype):
        """Initiate a new wrapper of `ftype` ('hankel' or 'fourier')."""

        # Store the type
        self._ftype = ftype
        self.available = list(FILTERS[ftype].keys())

        # Put all available filters as attributes
        for k, v in FILTERS[ftype].items():
            setattr(self, k, v)

    def __getattribute__(self, name):
        """Modify to load filter if the attribute is a know filter name."""

        # Get ftype
        ftype = object.__getattribute__(self, '_ftype')

        # If the `name` is in the corresponding dict => load filter
        if name in FILTERS[ftype].keys():

            # If filter is not yet cached, get it
            if FILTERS[ftype][name] is None:
                data = getattr(getattr(libdlf, ftype), name)
                dlf = DigitalFilter(name)
                for i, val in enumerate(['base', ] + data.values):
                    setattr(dlf, val, data()[i])
                dlf.factor = np.around([dlf.base[1]/dlf.base[0]], 15)

                # Cache it
                FILTERS[ftype][name] = dlf

            # Return filter
            return FILTERS[ftype][name]

        # Else, fall back to regular __getattribute__
        else:
            return object.__getattribute__(self, name)


class Hankel(_BaseFilter):
    """Wrapper to load Hankel-Transform filters from libdlf.

    You can either call a filter directly or first instantiate a Hankel object.
    Latter will give the possibility to explore the available filters with tab
    completion. A list of available filters is also stored in
    ``Hankel.available``.


    Examples
    --------

    .. ipython::

       In [1]: import empymod
          ...: dlf = empymod.filters.Hankel().wer_201_2018

       In [2]: H = empymod.filters.Hankel()
          ...: H.wer_201_2018.name
       Out[2]: 'wer_201_2018'

       In [3]: H.available
       Out[3]:
          ...: ['anderson_801_1982',
          ...:  'gupt_61_1997',
          ...:  'gupt_120_1997',
          ...:  'gupt_47_1997',
          ...:  'gupt_140_1997',
          ...:  'kong_61_2007b',
          ...:  'kong_121_2007',
          ...:  'kong_241_2007',
          ...:  'key_101_2009',
          ...:  'key_201_2009',
          ...:  'key_401_2009',
          ...:  'key_51_2012',
          ...:  'key_101_2012',
          ...:  'key_201_2012',
          ...:  'wer_201_2018',
          ...:  'wer_2001_2018']

    """

    def __init__(self):
        super().__init__('hankel')


class Fourier(_BaseFilter):
    """Wrapper to load Fourier-Transform filters from libdlf.

    You can either call a filter directly or first instantiate a Fourier
    object. Latter will give the possibility to explore the available filters
    with tab completion. A list of available filters is also stored in
    ``Fourier.available``.


    Examples
    --------

    .. ipython::

       In [1]: import empymod
          ...: dlf = empymod.filters.Fourier().key_201_2012

       In [2]: F = empymod.filters.Fourier()
          ...: F.key_201_2012.name
       Out[2]: 'key_201_2012'

       In [3]: F.available
          ...: ['key_81_2009',
          ...:  'key_241_2009',
          ...:  'key_601_2009',
          ...:  'key_101_2012',
          ...:  'key_201_2012',
          ...:  'grayver_50_2021']

    """

    def __init__(self):
        super().__init__('fourier')


# DEPRECATIONS - REMOVE in v3.0 #

def _deprecate_filter(func):
    """Decorator to deprecate filter functions."""

    def newfn():
        """Inner wrapper."""

        name = func.__name__

        # Check if Fourier or Hankel, adjust new name accordingly.
        if 'CosSin' in name:
            new = name.replace('_CosSin', '')
            ftype = 'Fourier'
        else:
            ftype = 'Hankel'
            new = name

        # Kong 61 has a different name in libdlf than here.
        if 'kong_61' in name:
            new += 'b'

        # Throw warning.
        msg = (
            f"Calling `empymod.filters.{name}()` is deprecated and will be "
            f"removed in v3.0; use `empymod.filters.{ftype}().{new}`."
        )
        warnings.warn(msg, DeprecationWarning)

        return func()

    return newfn


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
