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
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  See the
# License for the specific language governing permissions and limitations under
# the License.

# Import all modules
from . import model
from . import utils
from . import filters
from . import transform
from . import kernel
from . import scripts

# Import all functions, except the filters
from .model import *      # noqa  maint. in model.__all__
from .model import bipole, dipole, loop
from .utils import *      # noqa  maint. in utils.__all__
from .utils import EMArray, set_minimum, get_minimum
from .filters import DigitalFilter
from .transform import *  # noqa  maint. in transform.__all__
from .kernel import *     # noqa  maint. in kernel.__all__
from .scripts import *    # noqa  maint. in scripts.__init__.__all__

# Make only a selection available to __all__ to not clutter the namespace
# Maybe also to discourage the use of `from empymod import *`.
__all__ = ['model', 'utils', 'filters', 'transform', 'kernel', 'scripts',
           'bipole', 'dipole', 'loop', 'EMArray', 'set_minimum', 'get_minimum',
           'DigitalFilter']

# Version
try:
    # - Released versions just tags: v1.10.0;
    # - GitHub commits add 'dev#'+hash: 1.10.1.dev3+g973038c;
    # - Uncommited changes add timestamp: 1.10.1.dev3+g973038c.d20191022.
    from .version import version as __version__
except ImportError:
    # If used without using setup.py or without the .git-directory, or using
    # the Git or Zenodo zip-files or other unthought of ways of using it, we
    # provide here the last stable released version number, with a '-hc' for
    # hard-coded.
    # So v1.10.0-hc is most likely v1.10.0, but it can be anything between
    # v1.10.0 and v1.10.1.
    __version__ = '1.10.1-hc'
