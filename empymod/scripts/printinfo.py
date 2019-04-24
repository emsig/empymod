r"""
:mod:`printinfo` -- Tools to print date, time, and version information
======================================================================

Print or return date, time, and package version information in any environment
(Jupyter notebook, IPython console, Python console, QT console), either as
html-table (notebook) or as plain text (anywhere).

This script was heavily inspired by

- ``ipynbtools.py`` from https://github.com/qutip, and
- ``watermark.py`` from https://github.com/rasbt/watermark,

Always shown are the OS, number of CPU(s), ``numpy``, ``scipy``, ``empymod``,
``sys.version``, and time/date.

Additionally shown are, if they can be imported, ``IPython``, ``matplotlib``,
and ``numexpr``. It also shows MKL information, if available.

All modules provided in ``add_pckg`` are also shown. They have to be imported
before ``versions`` is called.

"""
# Copyright 2016-2019 Dieter Werthmüller
#
# This file is part of empymod.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License.  You may obtain a copy
# of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  See the
# License for the specific language governing permissions and limitations under
# the License.

# Mandatory modules
import sys
import time
import numpy
import scipy
import textwrap
import platform
import warnings
import multiprocessing

# empymod
import empymod

# Optional modules
try:
    import IPython
except ImportError:
    IPython = False
try:
    import matplotlib
except ImportError:
    matplotlib = False
try:
    import numexpr
except ImportError:
    numexpr = False
try:
    import mkl
except ImportError:
    mkl = False

# Get mkl info from numexpr or mkl, if available
if mkl:
    mklinfo = mkl.get_version_string()
elif numexpr:
    mklinfo = numexpr.get_vml_version()
else:
    mklinfo = False

__all__ = ['Versions', 'versions']


class Versions:
    """Version information."""

    def __init__(self, add_pckg=None, ncol=4):
        r"""Print date, time, and version information.

        Print date, time, and package version information in any environment
        (Jupyter notebook, IPython console, Python console, QT console), either
        as html-table (notebook) or as plain text (anywhere).

        This script was heavily inspired by:

            - ipynbtools.py from qutip https://github.com/qutip
            - watermark.py from https://github.com/rasbt/watermark

        Parameters
        ----------
        add_pckg : packages, optional
            Package or list of packages to add to output information (must be
            imported beforehand).

        ncol : int, optional
            Number of package-columns in html table; only has effect if
            ``mode='HTML'`` or ``mode='html'``. Defaults to 3.


        Examples
        --------
        >>> import pytest
        >>> import dateutil
        >>> from empymod import Versions
        >>> Versions()                            # Default values
        >>> Versions(pytest)                      # Provide additional package
        >>> Versions([pytest, dateutil], ncol=5)  # Set nr of columns

        """
        self.add_pckg = add_pckg
        self.ncol = ncol

    def __repr__(self):
        """Plain text information."""

        # Width for text-version
        n = 54
        text = '\n' + n*'-' + '\n'

        # Date and time info as title
        text += time.strftime('  %a %b %d %H:%M:%S %Y %Z\n\n')

        # OS and CPUs
        text += '{:>15}'.format(platform.system())+' : OS\n'
        text += '{:>15}'.format(multiprocessing.cpu_count())+' : CPU(s)\n'

        # Loop over packages
        for pckg in self._get_packages(self.add_pckg):
            text += '{:>15} : {}\n'.format(pckg.__version__, pckg.__name__)

        # sys.version
        text += '\n'
        for txt in textwrap.wrap(sys.version, n-4):
            text += '  '+txt+'\n'

        # mkl version
        if mklinfo:
            text += '\n'
            for txt in textwrap.wrap(mklinfo, n-4):
                text += '  '+txt+'\n'

        # Finish
        text += n*'-'

        return text

    def _repr_html_(self):
        """HTML-rendered versions information."""
        # Check ncol
        ncol = int(self.ncol)

        # Define html-styles
        border = "border: 2px solid #fff;'"

        def colspan(html, txt, ncol, nrow):
            r"""Print txt in a row spanning whole table."""
            html += "  <tr>\n"
            html += "     <td style='text-align: center; "
            if nrow == 0:
                html += "font-weight: bold; font-size: 1.2em; "
            elif nrow % 2 == 0:
                html += "background-color: #ddd;"
            html += border + " colspan='"
            html += str(2*ncol)+"'>%s</td>\n" % txt
            html += "  </tr>\n"
            return html

        def cols(html, version, name, ncol, i):
            r"""Print package information in two cells."""

            # Check if we have to start a new row
            if i > 0 and i % ncol == 0:
                html += "  </tr>\n"
                html += "  <tr>\n"

            html += "    <td style='text-align: right; background-color: "
            html += "#ccc; " + border + ">%s</td>\n" % version

            html += "    <td style='text-align: left; "
            html += border + ">%s</td>\n" % name

            return html, i+1

        # Start html-table
        html = "<table style='border: 3px solid #ddd;'>\n"

        # Date and time info as title
        html = colspan(html, time.strftime('%a %b %d %H:%M:%S %Y %Z'), ncol, 0)

        # OS and CPUs
        html += "  <tr>\n"
        html, i = cols(html, platform.system(), 'OS', ncol, 0)
        html, i = cols(html, multiprocessing.cpu_count(), 'CPU(s)', ncol, i)

        # Loop over packages
        for pckg in self._get_packages(self.add_pckg):
            html, i = cols(html, pckg.__version__, pckg.__name__, ncol, i)
        # Fill up the row
        while i % ncol != 0:
            html += "    <td style= " + border + "></td>\n"
            html += "    <td style= " + border + "></td>\n"
            i += 1
        # Finish row
        html += "  </tr>\n"

        # sys.version
        html = colspan(html, sys.version, ncol, 1)

        # mkl version
        if mklinfo:
            html = colspan(html, mklinfo, ncol, 2)

        # Finish table
        html += "</table>"

        return html

    @staticmethod
    def _get_packages(add_pckg):
        r"""Create list of packages."""

        # Mandatory packages
        pckgs = [numpy, scipy, empymod]

        # Optional packages
        for module in [IPython, numexpr, matplotlib]:
            if module:
                pckgs += [module]

        # Cast and add add_pckg
        if add_pckg is not None:

            # Cast add_pckg
            if isinstance(add_pckg, tuple):
                add_pckg = list(add_pckg)

            if not isinstance(add_pckg, list):
                add_pckg = [add_pckg, ]

            # Add add_pckg
            pckgs += add_pckg

        return pckgs


def versions(mode=None, add_pckg=None, ncol=4):
    r"""Old func-way of class `Versions`, here for backwards compatibility.

    ``mode`` is not used any longer, dummy here.
    """
    # Issue warning
    mesg = ("\n    Func `versions` is deprecated and will " +
            "be removed; use Class `Versions` instead.")
    warnings.warn(mesg, DeprecationWarning)

    return Versions(add_pckg, ncol)
