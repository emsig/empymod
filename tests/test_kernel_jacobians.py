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
        assert_allclose(Rm, val[2])

@pytest.mark.parametrize("njit", [False])
def test_reflections_jacobian(njit):                                    # 3. reflections
    if njit:
        reflections = kernel.reflections
    else:
        reflections = kernel.reflections.py_func
    depth = np.r_[-np.inf, 0, 5, 20, 40, 100, 140]  # Layer boundaries
    omega = 100
    eps = 8.854187817e-12
    mu_0 = 4 * np.pi * 1e-7
    kappa = 1
    cond = np.r_[1 / 2e14, 0.4, 1, 0.4, 1, 0.1, 0.01]

    e_zH = cond + 1j * omega * eps
    e_zH = e_zH.reshape(1, -1)
    z_eH = 1j * omega * mu_0 * np.ones_like(e_zH)
    gamma2 = z_eH * e_zH
    Gam = np.sqrt(kappa ** 2 + gamma2).reshape(1, 1, -1, 1)
    Rp, Rm, dRp, dRm = reflections(depth, e_zH, Gam, 0, 6, ana_deriv=True, z_eH=z_eH, debug=True)


    for cond_idx in np.arange(cond.size):
        h = 1e-6

        cond_pert = cond.copy()

        cond_pert[cond_idx] = cond_pert[cond_idx] + h
        e_zH = cond_pert + 1j * omega * eps
        e_zH = e_zH.reshape(1, -1)
        z_eH = 1j * omega * mu_0 * np.ones_like(e_zH)
        gamma2 = z_eH * e_zH
        Gam = np.sqrt(kappa ** 2 + gamma2).reshape(1, 1, -1, 1)
        Rp_pert, Rm_pert = reflections(depth, e_zH, Gam, 0, 6, ana_deriv=False, z_eH=z_eH, debug=True)

        cond_pert = cond.copy()
        cond_pert[cond_idx] = cond_pert[cond_idx] -  h
        e_zH = cond_pert + 1j * omega * eps
        e_zH = e_zH.reshape(1, -1)
        gamma2 = z_eH * e_zH
        Gam = np.sqrt(kappa ** 2 + gamma2).reshape(1, 1, -1, 1)
        Rp_pert2, Rm_pert2 = reflections(depth, e_zH, Gam, 0, 6, ana_deriv=False, z_eH=z_eH, debug=True)

        assert_allclose(dRp[0, 0, :, 0, cond_idx], (Rp_pert - Rp_pert2).flatten() / (2*h), rtol=1e-6,atol=1e-9,) #TODO: Is this strict enough?
        assert_allclose(dRm[0, 0, :, 0, cond_idx], (Rm_pert - Rm_pert2).flatten() / (2*h), rtol=1e-6,atol=1e-9,)

def test_all_dir():
    assert set(kernel.__all__) == set(dir(kernel))
