# kernel. Status: 2/7
import numpy as np
from os.path import join, dirname
from numpy.testing import assert_allclose

from empymod import kernel

# No input checks are carried out in kernel, by design. Input checks are
# carried out in model/utils, not in the core functions kernel/transform.
# Rubbish in, rubbish out. So we also do not check these functions for wrong
# inputs.

# Load required data
# Data generated with create_empymod.py
DATAEMPYMOD = np.load(join(dirname(__file__), 'data_empymod.npz'))
# Data generated with create_kernel.py
DATAKERNEL = np.load(join(dirname(__file__), 'data_kernel.npz'))

# 1. wavenumber

# 2. greenfct

# 3. reflections

# 4. fields


def test_angle_factor():                                      # 5. angle_factor
    dat = DATAKERNEL['angres'][()]
    for ddat in dat:
        res = kernel.angle_factor(**ddat['inp'])
        assert_allclose(res, ddat['res'])


def test_fullspace():                                            # 6. fullspace
    # Compare all to maintain status quo.
    fs = DATAEMPYMOD['fs'][()]
    fsres = DATAEMPYMOD['fsres'][()]
    for key in fs:
        # Get fullspace
        fs_res = kernel.fullspace(**fs[key])
        # Check
        assert_allclose(fs_res, fsres[key])


def test_halfspace():                                            # 7. halfspace
    # Compare all to maintain status quo.
    hs = DATAEMPYMOD['hs'][()]
    hsres = DATAEMPYMOD['hsres'][()]
    for key in hs:
        # Get halfspace
        hs_res = kernel.halfspace(**hs[key])
        # Check
        assert_allclose(hs_res, hsres[key])
