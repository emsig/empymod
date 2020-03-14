"""
Electromagnetic modeller to model electric or magnetic responses due to a
three-dimensional electric or magnetic source in a layered-earth model with
vertical transverse isotropic (VTI) resistivity, VTI electric permittivity, and
VTI magnetic permeability, from very low frequencies (DC) to very high
frequencies (GPR). The calculation is carried out in the wavenumber-frequency
domain, and various Hankel- and Fourier-transform methods are included to
transform the responses into the space-frequency and space-time domains.
"""
# Copyright 2016-2020 The empymod Developers.
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

# Import most important functions for __all__
from empymod.filters import DigitalFilter
from empymod.model import bipole, dipole, loop
from empymod.utils import EMArray, set_minimum, get_minimum

# We import all (*) from the following modules.
# What is imported is maintained in each module in `__all__`.
from empymod.model import *      # noqa
from empymod.utils import *      # noqa
from empymod.kernel import *     # noqa
from empymod.scripts import *    # noqa
from empymod.transform import *  # noqa

# Make only a selection available to __all__ to not clutter the namespace
# Maybe also to discourage the use of `from empymod import *`.
__all__ = ['model', 'utils', 'filters', 'transform', 'kernel', 'scripts',
           'bipole', 'dipole', 'loop', 'EMArray', 'set_minimum', 'get_minimum',
           'DigitalFilter']

# Version is sorted out in utils, so we can easier use it within the package
# itself.
__version__ = utils.__version__
