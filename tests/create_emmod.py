"""Test data for test_model::TestBipole::test_emmod.

Model parameters

11 models $\times$ 34 ab's  $=$ 374 cases.

- ab : all 36 (34) possibilities
- Frequencies: $f = 1.25\,\text{Hz}$
  NOTE: it would be good to include further tests with much lower/higher freqs!
- X-coordinates: -6000:3000:3000 m
- Y-coordinates: -4000:2000:2000 m
- Scaling (Depths and Coordinates): 0.001 (100 Hz); 1.0 (1 Hz); 4.0 (0.01 Hz)
- Models:


| layer   | 00  | 02  | 04  | 20  | 22  | 24  | 40  | 42  | 44  | 13  | 31  |
|:-------:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 0       | s r | s   | s   |   r |     |     |   r |     |     |     |     |
| 1       |     |     |     |     |     |     |     |     |     | s   |   r |
| 2       |     |   r |     | s   | s r | s   |     |   r |     |     |     |
| 3       |     |     |     |     |     |     |     |     |     |   r | s   |
| 4       |     |     |   r |     |     |   r | s   | s   | s r |     |     |

- zsrc: lower layer-boundary - 250 m
- zrec: lower layer-boundary - 150 m

"""
import numpy as np

from create_external import emmod

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
srcn = [0, 0, 0, 2, 2, 2, 4, 4, 4, 1, 3]
recn = [0, 2, 4, 0, 2, 4, 0, 2, 4, 3, 1]

# Define rest of model
freq = 1.25
depth = np.array([0, 500, 1100, 1800, 2600])
res = np.array([1, 3, 10, .01, 100])
aniso = np.array([.5, 4, 2, 1.5, 3])
epermH = np.array([1, 80, 5, 10, 20])
epermV = np.array([1, 80, 5, 20, 10])
mpermH = np.array([1, 1, 50, 1, 1])
mpermV = np.array([1, 1, 1, 50, 10])

outdict = {}
# Loop over different scenarios (src/rec in different layers)
for i in np.arange(np.size(srcn)):

    # Get src and rec layer
    sr = srcn[i]
    rr = recn[i]

    # Loop over ab-configurations
    for ab in pab:
        if ab in [36, 63]:
            continue

        # Get model-name
        modname = str(ab)+'_'+str(sr)+str(rr)
        print(modname)

        # Source and receiver
        if ab % 10 == 1:
            src = [0, 0, (depth[sr] - 250), 0, 0]
            msrc = False
        elif ab % 10 == 2:
            src = [0, 0, (depth[sr] - 250), 90, 0]
            msrc = False
        elif ab % 10 == 3:
            src = [0, 0, (depth[sr] - 250), 0, 90]
            msrc = False
        elif ab % 10 == 4:
            src = [0, 0, (depth[sr] - 250), 0, 0]
            msrc = True
        elif ab % 10 == 5:
            src = [0, 0, (depth[sr] - 250), 90, 0]
            msrc = True
        elif ab % 10 == 6:
            src = [0, 0, (depth[sr] - 250), 0, 90]
            msrc = True

        if ab // 10 == 1:
            rec = [xx, yy, (depth[rr] - 150), 0, 0]
            mrec = False
        elif ab // 10 == 2:
            rec = [xx, yy, (depth[rr] - 150), 90, 0]
            mrec = False
        elif ab // 10 == 3:
            rec = [xx, yy, (depth[rr] - 150), 0, 90]
            mrec = False
        elif ab // 10 == 4:
            rec = [xx, yy, (depth[rr] - 150), 0, 0]
            mrec = True
        elif ab // 10 == 5:
            rec = [xx, yy, (depth[rr] - 150), 90, 0]
            mrec = True
        elif ab // 10 == 6:
            rec = [xx, yy, (depth[rr] - 150), 0, 90]
            mrec = True

        # Run EMmod
        out = emmod(dx, nx, dy, ny, src, rec, depth[:-1], res, freq, aniso,
                    epermV, epermH, mpermV, mpermH, ab)

        # Collect input for bipole
        inp = {'src': src,
               'rec': rec,
               'depth': depth[:-1],
               'res': res,
               'freqtime': freq,
               'aniso': aniso,
               'epermH': epermH,
               'epermV': epermV,
               'mpermH': mpermH,
               'mpermV': mpermV,
               'msrc': msrc,
               'mrec': mrec,
               'verb': 0}

        outdict[modname] = (inp, out)

# # Store data # #
np.savez_compressed('data_emmod.npz', res=outdict)
