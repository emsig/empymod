r"""Create data for kernel tests. Kernel tests are just securing status quo."""
import numpy as np
from copy import deepcopy
from scipy.constants import mu_0, epsilon_0
from empymod import kernel, filters

# All possible (ab, msrc, mrec) combinations
pab = (np.arange(1, 7)[None, :] + np.array([10, 20, 30])[:, None]).ravel()
iab = {}
for mrec in [False, True]:
    for ab in pab:
        if ab == 36:
            continue
        if ab % 10 > 3:
            msrc = True
        else:
            msrc = False
        if mrec:
            msrc = not msrc
        iab[ab] = (msrc, mrec)

# # A -- ANGLE # #

angres = []
angle = np.array([1., 2., 4., 5.])
for key, val in iab.items():
    inp = {'angle': angle, 'ab': key, 'msrc': val[0], 'mrec': val[1]}
    res = kernel.angle_factor(angle, key, val[0], val[1])
    angres.append({'inp': inp, 'res': res})

# # B -- WAVENUMBER # #

# Example: 6-layer model; source in second layer, receiver in last
freq = np.array([0.003, 2.5, 1e6])
res = np.array([3, .3, 10, 4, 3, 1])
aniso = np.array([1, .5, 3, 1, 2, 1])
epermH = np.array([80, 100, 3, 8, 1, 1])
epermV = np.array([100, 30, 1, 10, 68, 9])
mpermH = np.array([.5, 100, 30, 1, 30, 1])
mpermV = np.array([2, 1, 30, 9, 50, 1])
etaH = 1/res + np.outer(2j*np.pi*freq, epermH*epsilon_0)
etaV = 1/(res*aniso*aniso) + np.outer(2j*np.pi*freq, epermV*epsilon_0)
zetaH = np.outer(2j*np.pi*freq, mpermH*mu_0)
zetaV = np.outer(2j*np.pi*freq, mpermV*mu_0)
lambd = filters.key_51_2012().base/np.array([0.001, 1, 100, 10000])[:, None]
depth = np.array([-np.infty, 0, 150, 300, 500, 600])
inp1 = {'zsrc': np.array([100]),
        'zrec': np.array([650]),
        'lsrc': np.array(1),
        'lrec': np.array(5),
        'depth': depth,
        'etaH': etaH,
        'etaV': etaV,
        'zetaH': zetaH,
        'zetaV': zetaV,
        'lambd': lambd,
        'xdirect': False,
        'use_ne_eval': False}
wave = {}
for key, val in iab.items():
    res = kernel.wavenumber(ab=key, msrc=val[0], mrec=val[1], **inp1)
    wave[key] = (key, val[0], val[1], inp1, res)

# # C -- GREENFCT # #

# Standard example
inp2 = deepcopy(inp1)
# Source and receiver in same layer (last)
inp3 = deepcopy(inp1)
inp3['zsrc'] = np.array([610])
inp3['lsrc'] = np.array(5)
# Receiver in first layer
inp4 = deepcopy(inp1)
inp4['zrec'] = np.array([-30])
inp4['lrec'] = np.array(0)
green = {}
for key, val in iab.items():
    res1 = kernel.greenfct(ab=key, msrc=val[0], mrec=val[1], **inp2)
    res2 = kernel.greenfct(ab=key, msrc=val[0], mrec=val[1], **inp3)
    res3 = kernel.greenfct(ab=key, msrc=val[0], mrec=val[1], **inp4)

    green[key] = (key, val[0], val[1], inp2, res1, inp3, res2, inp4, res3)


# # D -- REFLECTIONS # #
refl = {}
# Standard example
Gam = np.sqrt((etaH/etaV)[:, None, :, None] *
              (lambd**2)[None, :, None, :] + (zetaH**2)[:, None, :, None])
inp5 = {'depth': depth,
        'e_zH': etaH,
        'Gam': Gam,
        'lrec': inp1['lrec'],
        'lsrc': inp1['lsrc'],
        'use_ne_eval': False}
Rp1, Rm1 = kernel.reflections(**inp5)
refl[0] = (inp5, Rp1, Rm1)
# Source and receiver in same layer, but not last
inp6 = {'depth': inp2['depth'],
        'e_zH': etaH,
        'Gam': Gam,
        'lrec': np.array(3),
        'lsrc': np.array(3),
        'use_ne_eval': False}
Rp2, Rm2 = kernel.reflections(**inp6)
refl[1] = (inp6, Rp2, Rm2)

# # E -- FIELDS # #
# Standard example
inp7 = {'depth': depth,
        'Rp': Rp1,
        'Rm': Rm1,
        'Gam': Gam,
        'lrec': inp5['lrec'],
        'lsrc': inp5['lsrc'],
        'zsrc': inp1['zsrc'],
        'use_ne_eval': False}
# Source and receiver in same layer, but not last
inp8 = {'depth': depth,
        'Rp': Rp2,
        'Rm': Rm2,
        'Gam': Gam,
        'lrec': inp6['lrec'],
        'lsrc': inp6['lsrc'],
        'zsrc': np.array([350]),
        'use_ne_eval': False}

# Source and receiver in same layer, but not last
Rp4, Rm4 = kernel.reflections(depth, etaH, Gam, np.array(5),
                              np.array(5), False)
inp10 = {'depth': depth,
         'Rp': Rp4,
         'Rm': Rm4,
         'Gam': Gam,
         'lrec': np.array(5),
         'lsrc': np.array(5),
         'zsrc': np.array([700]),
         'use_ne_eval': False}

# Receiver in first layer, source in last
Rp3, Rm3 = kernel.reflections(depth, etaH, Gam, np.array(0),
                              np.array(5), False)
inp9 = {'depth': depth,
        'Rp': Rp3,
        'Rm': Rm3,
        'Gam': Gam,
        'lrec': np.array(0),
        'lsrc': np.array(5),
        'zsrc': np.array([700]),
        'use_ne_eval': False}

# Source in first layer, receiver in last
Rp5, Rm5 = kernel.reflections(depth, etaH, Gam, np.array(5),
                              np.array(0), False)
inp11 = {'depth': depth,
         'Rp': Rp5,
         'Rm': Rm5,
         'Gam': Gam,
         'lrec': np.array(5),
         'lsrc': np.array(0),
         'zsrc': np.array([-30]),
         'use_ne_eval': False}


fields = {}
for TM in [False, True]:
    for ab in pab:
        if TM and ab in [16, 26]:
            continue
        elif not TM and ab in [13, 23, 31, 32, 33, 34, 35]:
            continue
        elif ab == 36:
            continue

        out1 = kernel.fields(ab=ab, TM=TM, **inp7)
        out2 = kernel.fields(ab=ab, TM=TM, **inp8)
        out3 = kernel.fields(ab=ab, TM=TM, **inp9)
        out4 = kernel.fields(ab=ab, TM=TM, **inp10)
        out5 = kernel.fields(ab=ab, TM=TM, **inp11)
        fields[ab] = (ab, TM, inp7, out1, inp8, out2, inp9, out3, inp10, out4,
                      inp11, out5)

# # F -- Store data # #
np.savez_compressed('../data/kernel.npz', angres=angres, wave=wave,
                    green=green, refl=refl, fields=fields)
