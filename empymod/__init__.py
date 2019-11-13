"""
Electromagnetic modeller to model electric or magnetic responses due to a
three-dimensional electric or magnetic source in a layered-earth model with
vertical transverse isotropic (VTI) resistivity, VTI electric permittivity, and
VTI magnetic permeability, from very low frequencies (DC) to very high
frequencies (GPR). The calculation is carried out in the wavenumber-frequency
domain, and various Hankel- and Fourier-transform methods are included to
transform the responses into the space-frequency and space-time domains.
"""
# Copyright 2016-2019 The empymod Developers.
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
from empymod import filters
from empymod import transform
from empymod import kernel
from empymod import scripts

# Import all functions, except the filters
from empymod.model import *      # noqa  maint. in model.__all__
from empymod.model import bipole, dipole, loop
from empymod.utils import *      # noqa  maint. in utils.__all__
from empymod.utils import EMArray, set_minimum, get_minimum
from empymod.filters import DigitalFilter
from empymod.transform import *  # noqa  maint. in transform.__all__
from empymod.kernel import *     # noqa  maint. in kernel.__all__
from empymod.scripts import *    # noqa  maint. in scripts.__init__.__all__

# Make only a selection available to __all__ to not clutter the namespace
# Maybe also to discourage the use of `from empymod import *`.
__all__ = ['model', 'utils', 'filters', 'transform', 'kernel', 'scripts',
           'bipole', 'dipole', 'loop', 'EMArray', 'set_minimum', 'get_minimum',
           'DigitalFilter']

# Version
try:
    # - Released versions just tags:       1.10.0
    # - GitHub commits add .dev#+hash:     1.10.1.dev3+g973038c
    # - Uncommitted changes add timestamp: 1.10.1.dev3+g973038c.d20191022
    from empymod.version import version as __version__
except ImportError:
    # If it was not installed, then we don't know the version. We could throw a
    # warning here, but this case *should* be rare. empymod should be installed
    # properly!
    from datetime import datetime
    __version__ = 'unknown-'+datetime.today().strftime('%Y%m%d')
