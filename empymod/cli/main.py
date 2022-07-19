"""
Entry point for the command-line interface (CLI).
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

import os
import sys
import argparse

from empymod import utils
from empymod.cli import run


def main(args=None):
    """Parsing command line inputs of CLI interface."""

    # If not explicitly called, catch arguments
    if args is None:
        args = sys.argv[1:]

    # Start CLI-arg-parser and define arguments
    parser = argparse.ArgumentParser(
        description="3D electromagnetic modeller for 1D VTI media."
    )

    # arg: Modelling routine name
    parser.add_argument(
        "routine",
        nargs="?",
        default="bipole",
        type=str,
        help=("name of the modelling routine; default is 'bipole'; "
              "possibilities: 'bipole', 'dipole', 'loop', 'analytical'.")
    )

    # arg: Input file name
    parser.add_argument(
        "-i", "--input",
        type=str,
        default='input.json',
        help="input file name; default is 'input.json'"
    )

    # arg: Output file name
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="output file name; default is 'None' (STDOUT)"
    )

    # arg: Report
    parser.add_argument(
        "--report",
        action="store_true",
        default=False,
        help="only display empymod report"
    )

    # arg: Version
    parser.add_argument(
        "--version",
        action="store_true",
        default=False,
        help="only display empymod version"
    )

    # Get command line arguments.
    args_dict = vars(parser.parse_args(args))

    # Exits without simulation.
    if args_dict.pop('version'):  # empymod version info.

        print(f"empymod v{utils.__version__}")
        return

    elif args_dict.pop('report'):  # empymod report.
        print(utils.Report())
        return

    elif len(sys.argv) == 1 and not os.path.isfile('input.json'):

        # If no arguments provided and ./input.json does not exist, print info.
        print(parser.description)
        version = utils.__version__
        print(f"=> Type `empymod --help` for more info (empymod v{version}).")
        return

    # Run simulation with given command line inputs.
    run.simulation(args_dict)


if __name__ == "__main__":
    sys.exit(main())
