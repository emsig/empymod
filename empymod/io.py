"""
Utility functions for writing and reading inputs and data.
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

import re
import os
import json
import time

import numpy as np


from empymod import utils, filters

__all__ = ["save_input", "load_input", "save_data", "load_data"]


def __dir__():
    return __all__


def save_input(fname, data, **kwargs):
    """Save input dict to disk.

    Save the input provided to an empymod modelling routine on disk.


    Parameters
    ----------
    fname : str
        File name with absolute or relative path including suffix, which
        defines the used data format. Implemented is currently only ``.json``.

    data : dict
        Dictionary containing the parameters with their corresponding values
        for an empymod modelling routine.

    kwargs : optional
        Passed through to the saving method.

    """

    # Ensure fname is absolute.
    fname = os.path.abspath(fname)

    # Save JSON
    if fname.endswith(".json"):

        # For brevity yet readability, we create our custom formatted json,
        # where each model parameter is on a new line.
        out = "{"
        for k, v in data.items():
            out += "\n  "
            out += json.dumps({k: v}, cls=_ComplexNumPyEncoder)[1:-1]
            out += ","
        out = out[:-1] + "\n}"

        # Write it to disk.
        with open(fname, "w") as f:
            f.write(out)

    # Unknown, throw error
    else:
        raise ValueError(f"Unknown extension '.{fname.split('.')[-1]}'.")


def load_input(fname):
    """Load input from file.


    Parameters
    ----------
    fname : str
        File name with absolute or relative path including suffix, which
        defines the used data format. Implemented is currently only ``.json``.


    Returns
    -------
    data : dict
        Dictionary containing the input that was stored in the file.

    """

    # Ensure fname is absolute.
    fname = os.path.abspath(fname)

    # Save JSON
    if fname.endswith(".json"):
        with open(fname, "r", encoding="utf-8") as f:
            data = json.load(f)

    # Unknown, throw error
    else:
        raise ValueError(f"Unknown extension '.{fname.split('.')[-1]}'.")

    return data


def save_data(fname, data, **kwargs):
    """Save results from empymod.


    Parameters
    ----------
    fname : str
        File name with absolute or relative path including suffix, which
        defines the used data format. Implemented are currently:

        - ``.txt``: Uses numpy to store data to a plain text file.
        - ``.json``: Uses json to store inputs to a plain text file.

    data : ndarray
        The output from an empymod modelling routine.
        Note: You must set ``squeeze=False`` when calling the modelling
        routine, to obtain a 3D array (in case any of ``src``, ``rec``, or
        ``freqtime`` has only one entry).

    info : str, default: ""
        Information (one-line) to put into the header.

    kwargs : optional
        Passed through to the saving method.

    """
    # Ensure the right dimensionality.
    if data.ndim != 3:
        raise ValueError(
            "Data must be 3D (nfreqtime, nrec, nsrc); provided dimensions:  "
            f"{data.ndim}. You can achieve this by providing "
            "``squeeze=False`` to the modelling routine."
        )

    # Ensure fname is absolute.
    fname = os.path.abspath(fname)

    # Collect meta information.
    shape = data.shape
    meta = {
        "date": f"{time.strftime('%a %b %d %H:%M:%S %Y %Z')}",
        "version": f"empymod v{utils.__version__}",
        "shape": str(shape),
        "dtype": str(data.dtype),
        "info": kwargs.pop("info", "")
    }

    # Save txt with NumPy.
    if fname.endswith(".txt"):

        # Define format (depends if complex).
        crfmt = "%+.18e"
        if np.iscomplexobj(data):
            crfmt += "%+.18ej"

        # Formatting and setting.
        fmt = (shape[2]*(f"{crfmt}, "))[:-2]
        settings = {"delimiter": ", ", "fmt": fmt, "encoding": "utf-8"}

        with open(fname, "w", encoding="utf-8") as f:

            # Write meta data.
            for k, v in meta.items():
                f.write(f"# {k}:{' '+v if v else ''}\n")

            # write data.
            np.savetxt(f, X=data.reshape((-1, shape[2])), header="data",
                       **{**settings, **kwargs})

    # Save JSON
    elif fname.endswith(".json"):

        with open(fname, "w", encoding="utf-8") as f:
            json.dump({**meta, 'data': data}, f, cls=_ComplexNumPyEncoder,
                      **{"indent": 2, **kwargs})

    # Unknown, throw error
    else:
        raise ValueError(f"Unknown extension '.{fname.split('.')[-1]}'.")


def load_data(fname):
    """Load results from empymod stored with ``save_data``.


    Parameters
    ----------
    fname : str
        File name with absolute or relative path including suffix, which
        defines the used data format. Implemented are currently:

        - ``.txt``: Plain text file, loaded with np.loadtxt;
        - ``.json``: JSON plain text file.


    Returns
    -------
    EM : EMArray, (nfreqtime, nrec, nsrc)
        EM data.

    """

    # Ensure fname is absolute.
    fname = os.path.abspath(fname)

    # Load txt with NumPy.
    if fname.endswith(".txt"):

        # Read header for shape and dtype.
        meta = {}
        with open(fname, "r") as f:
            for line in f:
                if "data" in line:
                    break
                (key, val) = line.split(':', maxsplit=1)
                meta[key.lstrip('# ')] = val.lstrip(' ').rstrip()
        strshape = re.split(r'\(|\)', meta['shape'])[1]
        shape = tuple(map(int, strshape.split(",")))

        args = {"delimiter": ",", "dtype": meta['dtype'], "encoding": "utf-8"}
        data = np.loadtxt(fname, **args).reshape(shape)

    # Load JSON
    elif fname.endswith(".json"):

        # Load data.
        with open(fname, "r", encoding="utf-8") as f:
            inpdat = json.load(f)

        # If complex, re-create complex data.
        data = np.array(inpdat['data'])
        if 'complex' in inpdat['dtype']:
            data = data[0, ...] + 1j*data[1, ...]

    # Unknown, throw error
    else:
        raise ValueError(f"Unknown extension '.{fname.split('.')[-1]}'.")

    return utils.EMArray(data)


class _ComplexNumPyEncoder(json.JSONEncoder):
    """Custom json-encoder for NumPy, including complex data."""

    def default(self, obj):
        """Check if complex or NumPy, else pass on."""

        # If complex, stack [real, imag].
        if np.iscomplexobj(obj):
            obj = np.stack([np.asarray(obj).real, np.asarray(obj).imag])

        # Convert NumPy integers
        if isinstance(obj, np.integer):
            return int(obj)
        # Convert NumPy floats
        if isinstance(obj, np.floating):
            return float(obj)
        # Convert NumPy booleans
        if isinstance(obj, np.bool_):
            return bool(obj)
        # Convert NumPy arrays (includes complex)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, filters.DigitalFilter):
            return obj.name

        # Let the base class default method raise the TypeError.
        return json.JSONEncoder.default(self, obj)
