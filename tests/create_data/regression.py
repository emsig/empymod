r"""1836 models for a regression test with status quo.

The model generation is closely related to ./emmod.py.

Model parameters

18 models x 34 ab's x 3 frequencies = 1836 cases.

- ab : all 36 (34) possibilities
- Frequencies: $f = 0.01, 1.0, 100.0\,\text{Hz}$
- X-coordinates: -9000:3000:9000 m
- Y-coordinates: -6000:2000:6000 m
- Scaling (Depths and Coordinates): 0.001 (100 Hz); 1.0 (1 Hz); 4.0 (0.01 Hz)
- Models:


|layer| 13 | 00 | 03 | 05 | 30 | 33 | 35 | 50 | 53 | 55 | 24 | 42 |
|-----|----|----|----|----|----|----|----|----|----|----|----|----|
|  0  |    |s  r|s   |s   |   r|    |    |   r|    |    |    |    |
|  1  |s   |    |    |    |    |    |    |    |    |    |    |    |
|  2  |    |    |    |    |    |    |    |    |    |    |s   |   r|
|  3  |   r|    |   r|    |s   |s  r|s   |    |   r|    |    |    |
|  4  |    |    |    |    |    |    |    |    |    |    |   r|s   |
|  5  |    |    |    |   r|    |    |   r|s   |s   |s  r|    |    |

=> 13 is a homogenous fullspace
=> for 00, 33, and 55 exist 3 cases: zsrc > zrec, zsrc < zrec, zsrc = zrec

- zsrc: lower layer-boundary - 100 m; if lsrc=lrec: [-100, -200, -150] m
- zrec: lower layer-boundary -  90 m; if lsrc=lrec: [-200, -100, -150] m
- Naming: ab_freq_srrr
"""
import numpy as np

from empymod import dipole


# Define coordinates
y = np.arange(-3, 4)*2000
x = np.arange(-3, 4)*3000

# Define power of frequencies and scaling
fpow = np.array([-2, 0, 2])
ffac = np.array([4, 1, .001])

# Define src-rec configs
pab = (np.arange(60)+11).reshape(6, 10)[:, :6].ravel()

# Define model - src and rec
srcn = [1, 0, 0, 0, 0, 0, 3, 3, 3, 3, 3, 5, 5, 5, 5, 5, 2, 4]
recn = [3, 0, 0, 0, 3, 5, 0, 3, 3, 3, 5, 0, 3, 5, 5, 5, 4, 2]

# Source and receiver depths
dfsw = 0
dfsrc = [-100, -100, -200, -150]
dfrec = [-90, -200, -100, -150]

# Define rest of model
depth = np.array([0, 500, 1100, 1800, 2600, 3500])
res = [1, 3, 10, .01, 100, 1]
aniso = [.5, 4, 2, 1.5, 3, .25]
epermH = [1, 80, 5, 10, 20, 3]
epermV = [1, 80, 5, 20, 10, 8]
mpermH = [1, 1, 50, 1, 1, 3]
mpermV = [1, 1, 1, 50, 10, 1]

# Initialize out
out = dict()

dfsw = 0
# Loop over different scenarios (src/rec in different layers)
for i in np.arange(np.size(srcn)):
    # Get src and rec layer, store in model
    sr = srcn[i]
    rr = recn[i]
    if dfsw == 3:
        dfsw = 0
    if sr == rr:
        dfsw += 1
    else:
        dfsw = 0

    # First run is a fullspace (all parameters are equal)
    # All others are the model as defined
    if sr == 1 and rr == 3:
        ires = np.size(res)*[5, ]
        ianiso = np.size(res)*[2, ]
        iepermH = np.size(res)*[40, ]
        iepermV = np.size(res)*[80, ]
        impermH = np.size(res)*[10, ]
        impermV = np.size(res)*[20, ]
    else:
        ires = res
        ianiso = aniso
        iepermH = epermH
        iepermV = epermV
        impermH = mpermH
        impermV = mpermV

    # Loop over ab-configurations
    for ab in pab:
        if ab in [36, 63]:
            continue

        # Loop over frequencies
        for ifr in np.arange(3):

            freq = fpow[ifr]  # Freq

            # Update model with frequency
            ifreq = 10**np.float(freq)

            scl = ffac[ifr]  # Scaling factor

            isrc = [0, 0, (depth[sr] + dfsrc[dfsw])*scl]
            irec = [np.repeat(x*scl, 7).reshape(-1, 7).ravel('F'),
                    np.repeat(y*scl, 7), (depth[rr] + dfrec[dfsw])*scl]

            # Get model-name
            modname = str(ab)+'_'+str(freq)+'_'+str(sr)+str(rr)

            # Run model
            inp = {'src': isrc,
                   'rec': irec,
                   'depth': depth[:-1]*scl,
                   'res': ires,
                   'aniso': ianiso,
                   'freqtime': ifreq,
                   'ab': ab,
                   'opt': 'parallel',
                   'epermH': iepermH,
                   'epermV': iepermV,
                   'mpermH': impermH,
                   'mpermV': impermV,
                   'verb': 0,
                   'xdirect': True}
            em = dipole(**inp)

            # Store input and result
            out[modname] = (inp, em)

# Store data
np.savez_compressed('../data/regression.npz', res=out)
