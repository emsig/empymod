import pytest
import numpy as np
from os.path import join, dirname
from numpy.testing import assert_allclose
from copy import deepcopy
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
def test_reflections(njit):  # 3. reflections
    if njit:
        reflections = kernel.reflections
    else:
        # Note the setting "nb.config.DISABLE_JIT = True" in the kernel.py file
        reflections = kernel.reflections
        # reflections = kernel.reflections.py_func

    dat = DATAKERNEL['refl'][()]
    for _, val in dat.items():
        Rp, Rm = reflections(**val[0], debug=False)
        assert_allclose(Rp, val[1])
        assert_allclose(Rm, val[2])

@pytest.mark.parametrize("njit", [False])
def test_reflections_jacobian(njit):  # 3. reflections
    if njit:
        reflections = kernel.reflections
    else:
        # Note the setting "nb.config.DISABLE_JIT = True" in the kernel.py file
        reflections = kernel.reflections
        # reflections = kernel.reflections.py_func

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
    dGam = z_eH / (2 * Gam) # THIS ONLY HOLDS FOR THE ISOTROPIC CASE
    Rp, Rm, dRp, dRm = reflections(depth, e_zH, Gam, 0, 6, ana_deriv=True, dGam=dGam, debug=True)

    for cond_idx in np.arange(cond.size):
        h = 1e-6

        cond_pert = cond.copy()

        cond_pert[cond_idx] = cond_pert[cond_idx] + h
        e_zH = cond_pert + 1j * omega * eps
        e_zH = e_zH.reshape(1, -1)
        z_eH = 1j * omega * mu_0 * np.ones_like(e_zH)
        gamma2 = z_eH * e_zH
        Gam = np.sqrt(kappa ** 2 + gamma2).reshape(1, 1, -1, 1)
        Rp_pert, Rm_pert = reflections(depth, e_zH, Gam, 0, 6, ana_deriv=False, debug=True)

        cond_pert = cond.copy()
        cond_pert[cond_idx] = cond_pert[cond_idx] - h
        e_zH = cond_pert + 1j * omega * eps
        e_zH = e_zH.reshape(1, -1)
        gamma2 = z_eH * e_zH
        Gam = np.sqrt(kappa ** 2 + gamma2).reshape(1, 1, -1, 1)
        Rp_pert2, Rm_pert2 = reflections(depth, e_zH, Gam, 0, 6, ana_deriv=False, debug=True)

        assert_allclose(dRp[0, 0, :, 0, cond_idx], (Rp_pert - Rp_pert2).flatten() / (2 * h), rtol=1e-6,
                        atol=1e-9, )  # TODO: Is this strict enough?
        assert_allclose(dRm[0, 0, :, 0, cond_idx], (Rm_pert - Rm_pert2).flatten() / (2 * h), rtol=1e-6, atol=1e-9, )

@pytest.mark.parametrize("njit", [False])
def test_fields(njit):                                              # 4. fields
    if njit:
        fields = kernel.fields
    else:
        # Note the setting "nb.config.DISABLE_JIT = True" in the kernel.py file
        fields = kernel.fields
        # fields = kernel.fields.py_func

    dat = DATAKERNEL['fields'][()]
    for _, val in dat.items():
        for i in [2, 4, 6, 8, 10]:
            ab = val[0]
            TM = val[1]
            Pu, Pd = fields(ab=ab, TM=TM, **val[i])
            assert_allclose(Pu, val[i+1][0])
            assert_allclose(Pd, val[i+1][1])

@pytest.mark.parametrize("njit", [False])
def test_fields_jacobian(njit):
    if njit:
        fields = kernel.fields
        reflections = kernel.reflections
    else:
        # Note the setting "nb.config.DISABLE_JIT = True" in the kernel.py file
        fields = kernel.fields
        reflections = kernel.reflections
        # fields = kernel.fields.py_func
        # reflections = kernel.reflections.py_func


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
    dGam = z_eH / (2 * Gam) # THIS ONLY HOLDS FOR THE ISOTROPIC CASE
    Rp, Rm, dRp, dRm = reflections(depth, e_zH, Gam, 3, 3, ana_deriv=True, dGam = dGam, debug=False)

    Pu, Pd, dPu, dPd  = fields(depth, Rp.copy(), Rm.copy(), Gam.copy(), lrec=3, lsrc=3, zsrc=22, ab=66, TM=True, ana_deriv=True, dRp=dRp.copy(), dRm=dRm.copy(), dGam=dGam.copy())

    for cond_idx in np.arange(cond.size):
        h = 1e-9

        cond_pert = cond.copy()

        cond_pert[cond_idx] = cond_pert[cond_idx] + h
        e_zH = cond_pert + 1j * omega * eps
        e_zH = e_zH.reshape(1, -1)
        z_eH = 1j * omega * mu_0 * np.ones_like(e_zH)
        gamma2 = z_eH * e_zH
        Gam = np.sqrt(kappa ** 2 + gamma2).reshape(1, 1, -1, 1)
        Rp_pert, Rm_pert = reflections(depth, e_zH, Gam, 3, 3, ana_deriv=False, debug=False)
        Pu_pert, Pd_pert = fields(depth, Rp_pert, Rm_pert, Gam, lrec=3, lsrc=3, zsrc=22, ab=66, TM=True, ana_deriv=False,)

        cond_pert = cond.copy()
        cond_pert[cond_idx] = cond_pert[cond_idx] - h
        e_zH = cond_pert + 1j * omega * eps
        e_zH = e_zH.reshape(1, -1)
        gamma2 = z_eH * e_zH
        Gam = np.sqrt(kappa ** 2 + gamma2).reshape(1, 1, -1, 1)
        Rp_pert2, Rm_pert2 = reflections(depth, e_zH, Gam, 3, 3, ana_deriv=False, debug=False)
        Pu_pert2, Pd_pert2 = fields(depth, Rp_pert2, Rm_pert2, Gam, lrec=3, lsrc=3, zsrc=22, ab=66, TM=True, ana_deriv=False,)
        print(dPu[0, 0, 0, cond_idx])
        print((Pu_pert - Pu_pert2).flatten() / (2 * h))
        assert_allclose(dPu[0, 0, 0, cond_idx], (Pu_pert - Pu_pert2).flatten() / (2 * h), rtol=1e-6,
                        atol=1e-9, )  # TODO: Is this strict enough?
        #TODO: Increase presicion of (Pu_pert - Pu_pert2).flatten() / (2 * h)
        assert_allclose(dPd[0, 0, 0, cond_idx], (Pd_pert - Pd_pert2).flatten() / (2 * h), rtol=1e-6, atol=1e-9, )

@pytest.mark.parametrize("njit", [False])
def test_greenfct(njit):                                          # 2. greenfct
    if njit:
        greenfct = kernel.greenfct
    else:
        greenfct = kernel.greenfct
        # greenfct = kernel.greenfct.py_func


    dat = DATAKERNEL['green'][()]
    for _, val in dat.items():
        #for i in [3, 5, 7]:
        for i in [5]:
            ab = val[0]
            print(ab)
            msrc = val[1]
            mrec = val[2]
            out = greenfct(ab=ab, msrc=msrc, mrec=mrec, **val[i], ana_deriv=False)
            assert_allclose(out[0], val[i+1][0])
            assert_allclose(out[1], val[i+1][1])

    """val = dat[2]
    #for i in [3, 5, 7]:
    for i in [5]:
        ab = val[0]
        print(ab)
        msrc = val[1]
        mrec = val[2]
        out = greenfct(ab=ab, msrc=msrc, mrec=mrec, **val[i], ana_deriv=False)
        assert_allclose(out[0][0,:,0], val[i+1][0][0,:,0])
        # assert_allclose(out[1], val[i+1][1])"""

def test_greenfct_jacobian():
    # Note the setting "nb.config.DISABLE_JIT = True" in the kernel.py file
    greenfct = kernel.greenfct

    dat = DATAKERNEL['green'][()]
    i = 5
    val = dat[5]
    ab = 11
    msrc = False
    mrec = False
    # print(val[3])
    val[i]['etaV'] = val[i]['etaH']
    val[i]['zetaV'] = val[i]['zetaH']
    val[i]['xdirect'] = True
    GTM, GTE, dGTM, dGTE = greenfct(ab=ab, msrc=msrc, mrec=mrec, **val[i], ana_deriv=True, debug=False)

    """
        Real part of etaH is conductivity, imaginary part is permittivity and frequency. So add a small perturbation to the real part. 
    """
    etaH = val[i]['etaH']

    for cond_idx in[val[i]['lsrc']]:#np.arange(etaH.shape[1]):
        h = 1e-6
        val_pert = deepcopy(val[i])
        val_pert['etaH'][:, cond_idx] = val_pert['etaH'][:, cond_idx] + h
        GTM1, GTE1, = greenfct(ab=ab, msrc=msrc, mrec=mrec, **val_pert, ana_deriv=False, debug=False)

        val_pert = deepcopy(val[i])
        val_pert['etaH'][:, cond_idx] = val_pert['etaH'][:, cond_idx] - h
        GTM2, GTE2, = greenfct(ab=ab, msrc=msrc, mrec=mrec, **val_pert, ana_deriv=False, debug=False)

        assert_allclose(dGTM[0, :, 0, cond_idx], ((GTM1 - GTM2)/(2 * h))[0,:,0], rtol=1e-6,atol=1e-9, )
        # assert_allclose(dGTE[:, :, :, cond_idx], (GTE1 - GTE2)/(2 * h), rtol=1e-6,atol=1e-9, )

def test_greenfct_jacobian_debug():
    # Note the setting "nb.config.DISABLE_JIT = True" in the kernel.py file
    greenfct = kernel.greenfct

    dat = DATAKERNEL['green'][()]
    i = 5
    val = dat[5]
    ab = 11
    msrc = False
    mrec = False
    zsrc = val[i]['zsrc']
    zrec = val[i]['zrec']
    lsrc = val[i]['lsrc']
    lrec = val[i]['lrec']
    depth = val[i]['depth']
    etaH = val[i]['etaH']
    zetaH = val[i]['zetaH']
    lambd = val[i]['lambd']
    xdirect = val[i]['xdirect']
    etaV = etaH
    zetaV = zetaH

    gTM, gTE, dgTM, dgTE, gamTM, gamTE, dgamTM, dgamTE  = greenfct(zsrc, zrec, lsrc, lrec, depth, etaH, etaV, zetaH, zetaV, lambd,
             ab, xdirect, msrc, mrec, ana_deriv= True, debug = True)

    """
    Due to numerical instabilities, the perturbation is not exactly the same as the analytical derivative. Varying h can help to reduce the difference, but for different scales.
    Large lambdas -> larger h
    """

    """
        Real part of etaH is conductivity, imaginary part is permittivity and frequency. So add a small perturbation to the real part. 
    """
    # print(val[i])
    h = 1e-4
    for cond_idx in np.arange(etaH.shape[1]):
        etaH_pert = deepcopy(etaH)
        etaH_pert[:, cond_idx] = etaH_pert[:, cond_idx] + 2*h
        etaV_pert = etaH_pert
        gTM1, gTE1, gamTM1, gamTE1, = greenfct(zsrc, zrec, lsrc, lrec, depth, etaH_pert, etaV_pert, zetaH, zetaV, lambd,
         ab, xdirect, msrc, mrec, ana_deriv= False, debug = True)

        etaH_pert = deepcopy(etaH)
        etaH_pert[:, cond_idx] = etaH_pert[:, cond_idx] + h
        etaV_pert = etaH_pert
        gTM2, gTE2, gamTM2, gamTE2, = greenfct(zsrc, zrec, lsrc, lrec, depth, etaH_pert, etaV_pert, zetaH, zetaV, lambd,
         ab, xdirect, msrc, mrec, ana_deriv= False, debug = True)

        etaH_pert = deepcopy(etaH)
        etaH_pert[:, cond_idx] = etaH_pert[:, cond_idx] - h
        etaV_pert = etaH_pert
        gTM3, gTE3, gamTM3, gamTE3, = greenfct(zsrc, zrec, lsrc, lrec, depth, etaH_pert, etaV_pert, zetaH, zetaV, lambd,
            ab, xdirect, msrc, mrec, ana_deriv= False, debug = True)

        etaH_pert = deepcopy(etaH)
        etaH_pert[:, cond_idx] = etaH_pert[:, cond_idx] - 2*h
        etaV_pert = etaH_pert
        gTM4, gTE4, gamTM4, gamTE4, = greenfct(zsrc, zrec, lsrc, lrec, depth, etaH_pert, etaV_pert, zetaH, zetaV, lambd,
            ab, xdirect, msrc, mrec, ana_deriv= False, debug = True)



        # Gamma
        assert_allclose(dgamTM[:,:,cond_idx,:20], ((-gamTM1 + 8*gamTM2 - 8*gamTM3 + gamTM4)/(12*h))[:,:,cond_idx,:20], rtol=1e-6,atol=1e-8, )
        assert_allclose(dgamTE[:,:,cond_idx,:20], ((-gamTE1 + 8*gamTE2 - 8*gamTE3 + gamTE4)/(12*h))[:,:,cond_idx,:20], rtol=1e-6,atol=1e-8, )

        # Green's function
        assert_allclose(dgTM[0,:,:20,cond_idx], ((-gTM1 + 8*gTM2 - 8*gTM3 + gTM4)/(12*h))[0,:,:20], rtol=1e-6,atol=1e-8, )
        assert_allclose(dgTE[0,0,0,cond_idx], ((-gTE1 + 8*gTE2 - 8*gTE3 + gTE4)/(12*h))[0,0,0], rtol=1e-6,atol=1e-8, )




def test_all_dir():
    assert set(kernel.__all__) == set(dir(kernel))
