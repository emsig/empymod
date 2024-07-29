import os
import ctypes as ct

import numpy as np


cwavenumber = np.ctypeslib.load_library(
    "wavenumber", os.path.dirname(os.path.realpath(__file__))
)

cgreenfct = np.ctypeslib.load_library(
    "greenfct", os.path.dirname(os.path.realpath(__file__))
)

creflections = np.ctypeslib.load_library(
    "reflections", os.path.dirname(os.path.realpath(__file__))
)

cfields = np.ctypeslib.load_library(
    "fields", os.path.dirname(os.path.realpath(__file__))
)


def wavenumber(zsrc, zrec, lsrc, lrec, depth, etaH, etaV, zetaH, zetaV, lambd,
               ab, xdirect, msrc, mrec):

    c_doublep = ct.POINTER(ct.c_double)
    nfreq, nlayer = etaH.shape
    noff, nlambda = lambd.shape
    PJ0 = np.zeros((nfreq, noff, nlambda), dtype=complex)
    PJ1 = np.zeros((nfreq, noff, nlambda), dtype=complex)
    PJ0b = np.zeros((nfreq, noff, nlambda), dtype=complex)
    cwavenumber.wavenumber(
        int(nfreq), int(noff), int(nlayer), int(nlambda),
        ct.c_double(zsrc), ct.c_double(zrec),
        int(lsrc), int(lrec),
        depth.ctypes.data_as(c_doublep),
        etaH.ctypes.data_as(c_doublep),
        etaV.ctypes.data_as(c_doublep),
        zetaH.ctypes.data_as(c_doublep),
        zetaV.ctypes.data_as(c_doublep),
        lambd.ctypes.data_as(c_doublep),
        int(ab), int(xdirect), int(msrc), int(mrec),
        PJ0.ctypes.data_as(c_doublep),
        PJ1.ctypes.data_as(c_doublep),
        PJ0b.ctypes.data_as(c_doublep)
    )

    # Collect the output
    if ab not in [11, 22, 24, 15, 33]:
        PJ0 = None
    if ab not in [11, 12, 21, 22, 14, 24, 15, 25]:
        PJ0b = None
    if ab in [33, ]:
        PJ1 = None

    return PJ0, PJ1, PJ0b


def greenfct(zsrc, zrec, lsrc, lrec, depth, etaH, etaV, zetaH, zetaV, lambd,
             ab, xdirect, msrc, mrec, **kwargs):

    c_doublep = ct.POINTER(ct.c_double)
    nfreq, nlayer = etaH.shape
    noff, nlambda = lambd.shape

    GTM = np.zeros((nfreq, noff, nlambda), dtype=complex)
    GTE = np.zeros((nfreq, noff, nlambda), dtype=complex)
    cgreenfct.greenfct(
        int(nfreq), int(noff), int(nlayer), int(nlambda),
        ct.c_double(zsrc), ct.c_double(zrec), 
        int(lsrc), int(lrec), 
        depth.ctypes.data_as(c_doublep), 
        etaH.ctypes.data_as(c_doublep), 
        etaV.ctypes.data_as(c_doublep), 
        zetaH.ctypes.data_as(c_doublep), 
        zetaV.ctypes.data_as(c_doublep), 
        lambd.ctypes.data_as(c_doublep), 
        int(ab), int(xdirect), int(msrc), int(mrec),
        GTM.ctypes.data_as(c_doublep), 
        GTE.ctypes.data_as(c_doublep)
    )

    return GTM, GTE


def reflections(depth, e_zH, Gam, lrec, lsrc, **kwargs):
    c_doublep = ct.POINTER(ct.c_double)
    nfreq, noff, nlayer, nlambda = Gam.shape


    maxl = max([lrec, lsrc])
    minl = min([lrec, lsrc])
    nl = maxl-minl+1

    Rp = np.zeros((nfreq, noff, nl, nlambda), dtype=complex)
    Rm = np.zeros((nfreq, noff, nl, nlambda), dtype=complex)
    creflections.reflections(
        int(nfreq), int(noff), int(nlayer), int(nlambda),
        depth.ctypes.data_as(c_doublep), 
        e_zH.ctypes.data_as(c_doublep), 
        Gam.ctypes.data_as(c_doublep), 
        int(lrec),
        int(lsrc), 
        Rp.ctypes.data_as(c_doublep), 
        Rm.ctypes.data_as(c_doublep)
    )
    return Rp, Rm


def fields(depth, Rp, Rm, Gam, lrec, lsrc, zsrc, ab, TM):

    nfreq, noff, nlayer, nlambda = Rp.shape
    Pu = np.zeros((nfreq, noff, nlambda), dtype=complex)
    Pd = np.zeros((nfreq, noff, nlambda), dtype=complex)
    c_doublep = ct.POINTER(ct.c_double)
    cfields.fields(
        int(nfreq), int(noff), int(nlayer), int(nlambda),
        depth.ctypes.data_as(c_doublep), Rp.ctypes.data_as(c_doublep),
        Rm.ctypes.data_as(c_doublep), Gam.ctypes.data_as(c_doublep), int(lrec),
        int(lsrc), ct.c_double(zsrc), int(ab), int(TM),
        Pu.ctypes.data_as(c_doublep), Pd.ctypes.data_as(c_doublep)
    )

    return Pu, Pd
