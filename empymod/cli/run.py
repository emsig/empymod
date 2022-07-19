"""
Functions that actually call empymod within the CLI interface.
"""
# Copyright 2016-2022 The emsig community.
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

from empymod import model, io


def simulation(args_dict):
    """Run `empymod` invoked by CLI.

    Run ``empymod`` given the settings passed in ``args_dict`` (which
    correspond to command-line arguments).


    Parameters
    ----------
    args_dict : dict
        Arguments from terminal, see :func:`empymod.cli.main`.

    """

    # Checks.
    routines = ['bipole', 'dipole', 'loop', 'analytical']
    routine = args_dict['routine']
    if routine not in routines:
        raise ValueError(
            f"Routine must be one of {routines}; provided: '{routine}'."
        )

    # Get routine and load input.
    fct = getattr(model, routine)
    inpdat = io.load_input(args_dict['input'])

    # Run empymod, enforce ``squeeze=False``.
    out = fct(**{**inpdat, 'squeeze': False})

    # Store or print result.
    outfile = args_dict.pop('output')
    if outfile:
        info = f"Generated with <empymod.{args_dict['routine']}()> from "
        info += f"input <{args_dict['input']}>."
        io.save_data(outfile, out, info=info)
    else:
        print(out)
