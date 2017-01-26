# kernel. Status: 2/7
import numpy as np
from os.path import join, dirname
from numpy.testing import assert_allclose

from empymod.kernel import fullspace, halfspace

# No input checks are carried out in kernel, by design. Input checks are
# carried out in model/utils, not in the core functions kernel/transform.
# Rubbish in, rubbish out. So we also do not check these functions for wrong
# inputs.

# Load required data
# Data generated with create_empymod.py [25/01/2017]
DATAEMPYMOD = np.load(join(dirname(__file__), 'data_empymod.npz'))

# 1. wavenumber

# 2. angle_factor


def test_fullspace():                                            # 3. fullspace
    # Compare all to maintain status quo.
    fs = DATAEMPYMOD['fs'][()]
    fsres = DATAEMPYMOD['fsres'][()]
    for key in fs:
        # Get fullspace
        fs_res = fullspace(**fs[key])
        # Check
        assert_allclose(fs_res, fsres[key])


# 4. greenfct

# 5. reflections

# 6. fields


def test_halfspace():                                            # 7. halfspace
    # Compare all to maintain status quo.
    hs = DATAEMPYMOD['hs'][()]
    hsres = DATAEMPYMOD['hsres'][()]
    for key in hs:
        # Get halfspace
        hs_res = halfspace(**hs[key])
        # Check
        assert_allclose(hs_res, hsres[key])
