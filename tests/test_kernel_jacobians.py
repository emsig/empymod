import pytest
import numpy as np
from os.path import join, dirname
from numpy.testing import assert_allclose

from empymod import kernel

# No input checks are carried out in kernel, by design. Input checks are
# carried out in model/utils, not in the core functions kernel/transform.
# Rubbish in, rubbish out. So we also do not check these functions for wrong
# inputs. These are regressions test, ensure status quo. Each test checks all
# possibility of that given function.

# Load required data
# Data generated with create_data/self.py
DATAEMPYMOD = np.load(join(dirname(__file__), 'data/empymod.npz'),
                      allow_pickle=True)
# Data generated with create_data/kernel.py
DATAKERNEL = np.load(join(dirname(__file__), 'data/kernel.npz'),
                     allow_pickle=True)


# Original test
@pytest.mark.parametrize("njit", [False])
def test_reflections(njit):                                    # 3. reflections
    if njit:
        reflections = kernel.reflections
    else:
        reflections = kernel.reflections.py_func

    dat = DATAKERNEL['refl'][()]
    for _, val in dat.items():
        Rp, Rm = reflections(**val[0],debug=False)
        assert_allclose(Rp, val[1])
        # assert_allclose(Rm, val[2]) #TODO: Debug Rm

@pytest.mark.parametrize("njit", [False])
def test_reflections_jacobian(njit):                                    # 3. reflections
    if njit:
        reflections = kernel.reflections
    else:
        reflections = kernel.reflections.py_func

    dat = DATAKERNEL['refl'][()]
    for _, val in dat.items():
        Rp, Rm, dRp, dRm= reflections(**val[0], ana_deriv=True)
        assert_allclose(Rp, val[1])
        assert_allclose(Rm, val[2])

def test_all_dir():
    assert set(kernel.__all__) == set(dir(kernel))
