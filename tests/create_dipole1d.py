"""Create data for test_model::test_dipole1d."""
import numpy as np
from create_external import dipole1d

# # Comparison to DIPOLE1D for EE/ME

# Define model
freq = np.array([0.1])
depth = np.array([0, 200, 500, 1000])
res = np.array([1e20, 1/3, 1, 50, 1])
rec = [np.arange(1, 11)*1000, np.arange(-4, 6)*100, 195]


def collect_model(src, rec, freq, depth, res, srcpts=1):
    model = {'src': src,
             'rec': rec,
             'depth': depth,
             'res': res,
             'freq': freq,
             'srcpts': srcpts,
             'verb': 0}
    return model

# 1. x-directed dipole
src1 = [0, 0, 150, 0, 0]
out1 = dipole1d(src1, rec, depth, res, freq)
xdirdip = (collect_model(src1, rec, freq, depth, res), out1)

# 2. y-directed dipole
src2 = [0, 0, 150, 90, 0]
out2 = dipole1d(src2, rec, depth, res, freq)
ydirdip = (collect_model(src2, rec, freq, depth, res), out2)

# z-directed dipole
src3 = [0, 0, 150, 0, 90]
out3 = dipole1d(src3, rec, depth, res, freq)
zdirdip = (collect_model(src3, rec, freq, depth, res), out3)

# Dipole in xy-plane
src4 = [0, 0, 150, 23.5, 0]
out4 = dipole1d(src4, rec, depth, res, freq)
xydirdip = (collect_model(src4, rec, freq, depth, res), out4)

# Dipole in xz-plane
src5 = [0, 0, 150, 0, 39.6]
out5 = dipole1d(src5, rec, depth, res, freq)
xzdirdip = (collect_model(src5, rec, freq, depth, res), out5)

# Dipole in yz-plane
src6 = [0, 0, 150, 90, 69.6]
out6 = dipole1d(src6, rec, depth, res, freq)
yzdirdip = (collect_model(src6, rec, freq, depth, res), out6)

# Arbitrary xyz-dipole
src7 = [0, 0, 150, 13, 76]
out7 = dipole1d(src7, rec, depth, res, freq)
xyzdirdip = (collect_model(src7, rec, freq, depth, res), out7)

# x-directed bipole
src8 = [-50, 90, 0, 0, 150, 150]
out8 = dipole1d(src8, rec, depth, res, freq)
xdirbip = (collect_model(src8, rec, freq, depth, res), out8)

# y-directed bipole
src9 = [0, 0, -50, 90, 150, 150]
out9 = dipole1d(src9, rec, depth, res, freq)
ydirbip = (collect_model(src9, rec, freq, depth, res), out9)

# z-directed bipole
src10 = [0, 0, 0, 0, 140, 170]
out10 = dipole1d(src10, rec, depth, res, freq)
zdirbip = (collect_model(src10, rec, freq, depth, res), out10)

# # Store data # #
np.savez_compressed('data_dipole1d.npz',
                    xdirdip=xdirdip, ydirdip=ydirdip, zdirdip=zdirdip,
                    xydirdip=xydirdip, xzdirdip=xzdirdip, yzdirdip=yzdirdip,
                    xyzdirdip=xyzdirdip,
                    xdirbip=xdirbip, ydirbip=ydirbip, zdirbip=zdirbip,
                    )
