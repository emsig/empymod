r"""Test data for test_model::TestBipole::test_emmod.

Model parameters

11 models x 34 ab's = 374 cases.

- ab : all 36 (34) possibilities
- Frequencies: f = 0.013, 1.25, 130 Hz
               (just one freq per model, looping over it)
- Models:

| layer   | 00  | 02  | 04  | 20  | 22  | 24  | 40  | 42  | 44  | 13  | 31  |
|:-------:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 0       | s r | s   | s   |   r |     |     |   r |     |     |     |     |
| 1       |     |     |     |     |     |     |     |     |     | s   |   r |
| 2       |     |   r |     | s   | s r | s   |     |   r |     |     |     |
| 3       |     |     |     |     |     |     |     |     |     |   r | s   |
| 4       |     |     |   r |     |     |   r | s   | s   | s r |     |     |

"""
import numpy as np

from external import emmod

# 1. Define coordinates for empymod:

# We only check 8 distinct locations (faster)
dx = 3000
dy = 2000
nx = 4
ny = 4
xx = np.array([0, 3, 3, 3, -6, -6])*1000.
yy = np.array([2, 0, 2, -4, 2, -4])*1000.

# All possible ab's
pab = (np.arange(60)+11).reshape(6, 10)[:, :6].ravel()
# pab = (np.arange(60)+11).reshape(6, 10)[:3, :3].ravel()  # EE
# pab = (np.arange(60)+11).reshape(6, 10)[3:, :3].ravel()  # ME
# pab = (np.arange(60)+11).reshape(6, 10)[:3, 3:6].ravel()  # EM
# pab = (np.arange(60)+11).reshape(6, 10)[3:, 3:6].ravel()  # MM

# Define model - src and rec
fact = [2, 1, 0.01]
srcn = [0, 0, 0, 2, 2, 2, 4, 4, 4, 1, 3]
recn = [0, 2, 4, 0, 2, 4, 0, 2, 4, 3, 1]
srcm = [250, 250, 2.5]
recm = [150, 150, 1.5]

# Define rest of model
freqs = [0.013, 1.25, 130]
depths = [np.array([0, 1000, 2000, 3000, 6000]),
          np.array([0, 500, 1100, 1800, 2600]),
          np.array([0, 10, 20, 30, 60])]
res = np.array([1, 3, 10, .01, 100])
aniso = np.array([.5, 4, 2, 1.5, 3])
epermH = np.array([1, 80, 5, 10, 20])
epermV = np.array([1, 80, 5, 20, 10])
mpermH = np.array([1, 1, 50, 1, 1])
mpermV = np.array([1, 1, 1, 50, 10])

nam = ['_0-013', '_1-25', '_130']
kmax = [10, 10, 200]
maxpt = [1000, 1000, 2000]

outdict = {}
# Loop over different scenarios (src/rec in different layers)
ifr = 0
for i in np.arange(np.size(srcn)):

    # Get src and rec layer
    sr = srcn[i]
    rr = recn[i]

    # Loop over ab-configurations
    for ab in pab:
        if ab in [36, 63]:
            continue

        scl = fact[ifr]
        # Source and receiver
        if ab % 10 == 1:
            src = [0, 0, (depths[ifr][sr] - srcm[ifr]), 0, 0]
            msrc = False
        elif ab % 10 == 2:
            src = [0, 0, (depths[ifr][sr] - srcm[ifr]), 90, 0]
            msrc = False
        elif ab % 10 == 3:
            src = [0, 0, (depths[ifr][sr] - srcm[ifr]), 0, 90]
            msrc = False
        elif ab % 10 == 4:
            src = [0, 0, (depths[ifr][sr] - srcm[ifr]), 0, 0]
            msrc = True
        elif ab % 10 == 5:
            src = [0, 0, (depths[ifr][sr] - srcm[ifr]), 90, 0]
            msrc = True
        elif ab % 10 == 6:
            src = [0, 0, (depths[ifr][sr] - srcm[ifr]), 0, 90]
            msrc = True

        if ab // 10 == 1:
            rec = [xx*scl, yy*scl, (depths[ifr][rr] - recm[ifr]), 0, 0]
            mrec = False
        elif ab // 10 == 2:
            rec = [xx*scl, yy*scl, (depths[ifr][rr] - recm[ifr]), 90, 0]
            mrec = False
        elif ab // 10 == 3:
            rec = [xx*scl, yy*scl, (depths[ifr][rr] - recm[ifr]), 0, 90]
            mrec = False
        elif ab // 10 == 4:
            rec = [xx*scl, yy*scl, (depths[ifr][rr] - recm[ifr]), 0, 0]
            mrec = True
        elif ab // 10 == 5:
            rec = [xx*scl, yy*scl, (depths[ifr][rr] - recm[ifr]), 90, 0]
            mrec = True
        elif ab // 10 == 6:
            rec = [xx*scl, yy*scl, (depths[ifr][rr] - recm[ifr]), 0, 90]
            mrec = True

        # Run EMmod
        out = emmod(dx*scl, nx, dy*scl, ny, src, rec, depths[ifr][:-1], res,
                    freqs[ifr], aniso, epermV, epermH, mpermV, mpermH, ab,
                    kmax=kmax[ifr], maxpt=maxpt[ifr])

        # Collect input for bipole
        inp = {'src': src,
               'rec': rec,
               'depth': depths[ifr][:-1],
               'res': res,
               'freqtime': freqs[ifr],
               'aniso': aniso,
               'epermH': epermH,
               'epermV': epermV,
               'mpermH': mpermH,
               'mpermV': mpermV,
               'msrc': msrc,
               'mrec': mrec,
               'verb': 0}

        # Store model
        modname = str(ab)+'_'+str(sr)+str(rr)+nam[ifr]
        outdict[modname] = (inp, out)

        ifr += 1
        if ifr > 2:
            ifr = 0

# # Store data # #
np.savez_compressed('../data/emmod.npz', res=outdict)
