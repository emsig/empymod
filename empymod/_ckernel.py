import os
import ctypes as ct

import numpy as np


C_DOUBLEP = ct.POINTER(ct.c_double)
path = os.path.dirname(os.path.realpath(__file__))
cwavenumber = np.ctypeslib.load_library("wavenumber", path)
cgreenfct = np.ctypeslib.load_library("greenfct", path)
creflections = np.ctypeslib.load_library("reflections", path)
cfields = np.ctypeslib.load_library("fields", path)


def wavenumber(zsrc, zrec, lsrc, lrec, depth, etaH, etaV, zetaH, zetaV, lambd,
               ab, xdirect, msrc, mrec):
    """C-Wrapper for wavenumber."""
    nfreq, nlayer = etaH.shape
    noff, nlambda = lambd.shape
    PJ0 = np.zeros((nfreq, noff, nlambda), dtype=complex)
    PJ1 = np.zeros((nfreq, noff, nlambda), dtype=complex)
    PJ0b = np.zeros((nfreq, noff, nlambda), dtype=complex)
    cwavenumber.wavenumber(
        int(nfreq),
        int(noff),
        int(nlayer),
        int(nlambda),
        ct.c_double(zsrc),
        ct.c_double(zrec),
        int(lsrc),
        int(lrec),
        depth.ctypes.data_as(C_DOUBLEP),
        etaH.ctypes.data_as(C_DOUBLEP),
        etaV.ctypes.data_as(C_DOUBLEP),
        zetaH.ctypes.data_as(C_DOUBLEP),
        zetaV.ctypes.data_as(C_DOUBLEP),
        lambd.ctypes.data_as(C_DOUBLEP),
        int(ab),
        int(xdirect),
        int(msrc),
        int(mrec),
        PJ0.ctypes.data_as(C_DOUBLEP),
        PJ1.ctypes.data_as(C_DOUBLEP),
        PJ0b.ctypes.data_as(C_DOUBLEP),
    )
    if ab not in [11, 22, 24, 15, 33]:
        PJ0 = None
    if ab not in [11, 12, 21, 22, 14, 24, 15, 25]:
        PJ0b = None
    if ab in [33, ]:
        PJ1 = None
    return PJ0, PJ1, PJ0b


def greenfct(zsrc, zrec, lsrc, lrec, depth, etaH, etaV, zetaH, zetaV, lambd,
             ab, xdirect, msrc, mrec):
    """C-Wrapper for greenfct."""
    nfreq, nlayer = etaH.shape
    noff, nlambda = lambd.shape
    GTM = np.zeros((nfreq, noff, nlambda), dtype=complex)
    GTE = np.zeros((nfreq, noff, nlambda), dtype=complex)
    cgreenfct.greenfct(
        int(nfreq),
        int(noff),
        int(nlayer),
        int(nlambda),
        ct.c_double(zsrc),
        ct.c_double(zrec),
        int(lsrc),
        int(lrec),
        depth.ctypes.data_as(C_DOUBLEP),
        etaH.ctypes.data_as(C_DOUBLEP),
        etaV.ctypes.data_as(C_DOUBLEP),
        zetaH.ctypes.data_as(C_DOUBLEP),
        zetaV.ctypes.data_as(C_DOUBLEP),
        lambd.ctypes.data_as(C_DOUBLEP),
        int(ab),
        int(xdirect),
        int(msrc),
        int(mrec),
        GTM.ctypes.data_as(C_DOUBLEP),
        GTE.ctypes.data_as(C_DOUBLEP),
    )
    return GTM, GTE


def reflections(depth, e_zH, Gam, lrec, lsrc):
    """C-Wrapper for reflections."""
    nfreq, noff, nlayer, nlambda = Gam.shape
    maxl = max([lrec, lsrc])
    minl = min([lrec, lsrc])
    nl = maxl-minl+1
    Rp = np.zeros((nfreq, noff, nl, nlambda), dtype=complex)
    Rm = np.zeros((nfreq, noff, nl, nlambda), dtype=complex)
    creflections.reflections(
        int(nfreq),
        int(noff),
        int(nlayer),
        int(nlambda),
        depth.ctypes.data_as(C_DOUBLEP),
        e_zH.ctypes.data_as(C_DOUBLEP),
        Gam.ctypes.data_as(C_DOUBLEP),
        int(lrec),
        int(lsrc),
        Rp.ctypes.data_as(C_DOUBLEP),
        Rm.ctypes.data_as(C_DOUBLEP),
    )
    return Rp, Rm


def fields(depth, Rp, Rm, Gam, lrec, lsrc, zsrc, ab, TM):
    """C-Wrapper for fields."""
    nfreq, noff, nlayer, nlambda = Rp.shape
    Pu = np.zeros((nfreq, noff, nlambda), dtype=complex)
    Pd = np.zeros((nfreq, noff, nlambda), dtype=complex)
    cfields.fields(
        int(nfreq),
        int(noff),
        int(nlayer),
        int(nlambda),
        depth.ctypes.data_as(C_DOUBLEP),
        Rp.ctypes.data_as(C_DOUBLEP),
        Rm.ctypes.data_as(C_DOUBLEP),
        Gam.ctypes.data_as(C_DOUBLEP),
        int(lrec),
        int(lsrc),
        ct.c_double(zsrc),
        int(ab),
        int(TM),
        Pu.ctypes.data_as(C_DOUBLEP),
        Pd.ctypes.data_as(C_DOUBLEP),
    )
    return Pu, Pd
