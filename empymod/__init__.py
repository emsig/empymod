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

# Import all modules
from empymod import model
from empymod import utils
from empymod import kernel
from empymod import filters
from empymod import scripts
from empymod import transform

# Import most important functions
from empymod.filters import DigitalFilter
from empymod.model import bipole, dipole, loop
from empymod.utils import EMArray, set_minimum, get_minimum, Report

# For top-namespace
from empymod.scripts import fdesign, tmtemod
from empymod.model import analytical, gpr, dipole_k, fem, tem

__all__ = ['model', 'utils', 'filters', 'transform', 'kernel', 'scripts',
           'bipole', 'dipole', 'loop', 'EMArray', 'set_minimum', 'get_minimum',
           'DigitalFilter', 'Report']

# Version defined in utils, so we can easier use it within the package itself.
__version__ = utils.__version__
