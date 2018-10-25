r"""Create data for test_model::test_green3d."""
import numpy as np
from external import green3d

# # Comparison to Green3D for EE (electrical source, electrical receiver)

# Define model
freq = np.array([0.1])
depth = np.array([0, 200, 500, 1000])
res = np.array([1e20, 1/3, 1, 50, 1])
aniso = np.array([1, 1, 1.5, 2, 3])
rec = [np.arange(1, 11)*1000, np.arange(-4, 6)*100, 195]
srca = 349  # Source strength


def collect_model(src, rec, freq, depth, res, aniso, strength=0, srcpts=1,
                  msrc=False):
    model = {'src': src,
             'rec': rec,
             'depth': depth,
             'res': res,
             'freq': freq,
             'aniso': aniso,
             'strength': strength,
             'srcpts': srcpts,
             'msrc': msrc}
    return model


# 1. x-directed electric dipole
src1 = [0, 0, 150, 0, 0]
out1 = green3d(src1, rec, depth, res, freq, aniso, 9)
xdirdip = (collect_model(src1, rec, freq, depth, res, aniso), out1)

# 2. y-directed electric dipole
src2 = [0, 0, 150, 90, 0]
out2 = green3d(src2, rec, depth, res, freq, aniso, 9)
ydirdip = (collect_model(src2, rec, freq, depth, res, aniso), out2)

# 3. z-directed electric dipole
src3 = [0, 0, 150, 0, 90]
out3 = green3d(src3, rec, depth, res, freq, aniso, 9)
zdirdip = (collect_model(src3, rec, freq, depth, res, aniso), out3)

# 4. Electric dipole in xy-plane
src4 = [0, 0, 150, 23.5, 0]
out4 = green3d(src4, rec, depth, res, freq, aniso, 9)
xydirdip = (collect_model(src4, rec, freq, depth, res, aniso), out4)

# 5. Electric dipole in xz-plane
src5 = [0, 0, 150, 0, 39.6]
out5 = green3d(src5, rec, depth, res, freq, aniso, 9)
xzdirdip = (collect_model(src5, rec, freq, depth, res, aniso), out5)

# 6. Electric dipole in yz-plane
src6 = [0, 0, 150, 90, 69.6]
out6 = green3d(src6, rec, depth, res, freq, aniso, 9)
yzdirdip = (collect_model(src6, rec, freq, depth, res, aniso), out6)

# 7. Arbitrary electric xyz-dipole
src7 = [0, 0, 150, 13, 76]
out7 = green3d(src7, rec, depth, res, freq, aniso, 9)
xyzdirdip = (collect_model(src7, rec, freq, depth, res, aniso), out7)

# 8. x-directed electric bipole
src8 = [-50, 90, 0, 0, 150, 150]
out8 = green3d(src8, rec, depth, res, freq, aniso, 3, srca)
xdirbip = (collect_model(src8, rec, freq, depth, res, aniso, srca, 10), out8)

# 9. y-directed electric bipole
src9 = [0, 0, -50, 90, 150, 150]
out9 = green3d(src9, rec, depth, res, freq, aniso, 3, srca)
ydirbip = (collect_model(src9, rec, freq, depth, res, aniso, srca, 10), out9)

# 10. z-directed electric bipole
src10 = [0, 0, 0, 0, 140, 170]
out10 = green3d(src10, rec, depth, res, freq, aniso, 2, srca)
zdirbip = (collect_model(src10, rec, freq, depth, res, aniso, srca, 10), out10)

# 11. x-directed magnetic bipole
out11 = green3d(src1, rec, depth, res, freq, aniso, 6)
xdirbipm = (collect_model(src1, rec, freq, depth, res, aniso, 0, 10, True),
            out11)

# 12. y-directed magnetic bipole
out12 = green3d(src2, rec, depth, res, freq, aniso, 7)
ydirbipm = (collect_model(src2, rec, freq, depth, res, aniso, 0, 10, True),
            out12)

# 13. z-directed magnetic bipole
out13 = green3d(src3, rec, depth, res, freq, aniso, 8)
zdirbipm = (collect_model(src3, rec, freq, depth, res, aniso, 0, 10, True),
            out13)

# 14. x-directed magnetic dipole
out14 = green3d(src1, rec, depth, res, freq, aniso, 10)
xdirdipm = (collect_model(src1, rec, freq, depth, res, aniso, 0, 1, True),
            out14)

# 15. y-directed magnetic dipole
out15 = green3d(src2, rec, depth, res, freq, aniso, 10)
ydirdipm = (collect_model(src2, rec, freq, depth, res, aniso, 0, 1, True),
            out15)

# 16. z-directed magnetic dipole
out16 = green3d(src3, rec, depth, res, freq, aniso, 10)
zdirdipm = (collect_model(src3, rec, freq, depth, res, aniso, 0, 1, True),
            out16)

# 17. Magnetic dipole in xy-plane
out17 = green3d(src4, rec, depth, res, freq, aniso, 10)
xydirdipm = (collect_model(src4, rec, freq, depth, res, aniso, 0, 1, True),
             out17)

# 18. Magnetic dipole in xz-plane
out18 = green3d(src5, rec, depth, res, freq, aniso, 10)
xzdirdipm = (collect_model(src5, rec, freq, depth, res, aniso, 0, 1, True),
             out18)

# 19. Magnetic dipole in yz-plane
out19 = green3d(src6, rec, depth, res, freq, aniso, 10)
yzdirdipm = (collect_model(src6, rec, freq, depth, res, aniso, 0, 1, True),
             out19)

# 20. Arbitrary magnetic xyz-dipole
out20 = green3d(src7, rec, depth, res, freq, aniso, 10)
xyzdirdipm = (collect_model(src7, rec, freq, depth, res, aniso, 0, 1, True),
              out20)

# # Store data # #
np.savez_compressed('../data/green3d.npz',
                    xdirdip=xdirdip, ydirdip=ydirdip, zdirdip=zdirdip,
                    xydirdip=xydirdip, xzdirdip=xzdirdip, yzdirdip=yzdirdip,
                    xyzdirdip=xyzdirdip,
                    xdirbip=xdirbip, ydirbip=ydirbip, zdirbip=zdirbip,
                    xdirdipm=xdirdipm, ydirdipm=ydirdipm, zdirdipm=zdirdipm,
                    xydirdipm=xydirdipm, xzdirdipm=xzdirdipm,
                    yzdirdipm=yzdirdipm, xyzdirdipm=xyzdirdipm,
                    xdirbipm=xdirbipm, ydirbipm=ydirbipm, zdirbipm=zdirbipm,
                    )
