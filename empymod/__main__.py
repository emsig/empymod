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


import sys
import argparse

from empymod import io, model, utils


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
        type=str,
        choices=['bipole', 'dipole', 'loop', 'analytical'],
        help=("name of the modelling routine")
    )

    # arg: Input file name
    parser.add_argument(
        "input",
        nargs="?",
        type=str,
        help="input file name"
    )

    # arg: Output file name
    parser.add_argument(
        "output",
        nargs="?",
        default=None,
        type=str,
        help="output file name; prints to STDOUT if not provided"
    )

    # arg: Report
    parser.add_argument(
        "--report",
        action="store_true",
        default=False,
        help="show the empymod report and exit"
    )

    # arg: Version
    parser.add_argument(
        "--version",
        action="store_true",
        default=False,
        help="show the empymod version and exit"
    )

    # Get command line arguments.
    args_dict = vars(parser.parse_args(args))

    # empymod version info.
    if args_dict.pop('version'):  # empymod version info.
        print(f"empymod v{utils.__version__}")

    # empymod report.
    elif args_dict.pop('report'):
        print(utils.Report())

    # Info if not at list routine and input provided.
    elif len(sys.argv) < 3:
        print(f"{parser.description}\n=> Type `empymod --help` for "
              f"more info (empymod v{utils.__version__}).")

    # Actually compute.
    else:
        try:
            run(args_dict)
        except (AttributeError, TypeError, ValueError, FileNotFoundError) as e:
            return e


def run(args_dict):
    """Run empymod with provided arguments."""

    # Run empymod, enforce ``squeeze=False``.
    iname = args_dict['input']
    fct = args_dict['routine']
    out = getattr(model, fct)(**{**io.load_input(iname), 'squeeze': False})

    # Store or print result.
    outfile = args_dict.pop('output')
    if outfile:
        info = f"Generated with <empymod.{fct}()> from input <{iname}>."
        io.save_data(outfile, out, info=info)
    else:
        print(out)


if __name__ == "__main__":
    sys.exit(main())
