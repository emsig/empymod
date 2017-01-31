"""Create data for kernel tests. Kernel tests are just securing status quo."""
import numpy as np
from empymod import kernel


# # A -- ANGLE # #

angres = []
angle = np.array([1., 2., 4., 5.])
pab = np.arange(1, 7)[None, :] + np.array([10, 20, 30])[:, None]
for msrc in [True, False]:
    for mrec in [True, False]:
        for ab in pab.ravel():
            inp = {'angle': angle, 'ab': ab, 'msrc': msrc, 'mrec': mrec}
            res = kernel.angle_factor(angle, ab, msrc, mrec)
            angres.append({'inp': inp, 'res': res})

# # X -- Store data # #
np.savez_compressed('data_kernel.npz', angres=angres)
