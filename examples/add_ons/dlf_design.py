"""
Calculating a Digital Linear Filter.
====================================

This is an example for the add-on ``fdesign``. The example is taken from the
article Werthmüller et al., 2019. Have a look at the article repository on
`empymod/article-fdesign <https://github.com/empymod/article-fdesign>`_ for
many more examples.

**Reference**

- Werthmüller, D., K. Key, and E. Slob, 2019, **A tool for designing digital
  filters for the Hankel and Fourier transforms in potential, diffusive, and
  wavefield modeling**:  Geophysics, 84(2), F47-F56; DOI:
  `10.1190/geo2018-0069.1 <http://doi.org/10.1190/geo2018-0069.1>`_.

"""
import empymod
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')

###############################################################################

inp = {'r': np.logspace(0, 10, 1000),
       'r_def': (1, 1, 2),
       'n': 201,
       'name': 'test',
       'full_output': True,
       'fI': (empymod.fdesign.j0_1(5), empymod.fdesign.j1_1(5))}


###############################################################################
# 1. Rough overview over a wide range
# -----------------------------------

filt1, out1 = empymod.fdesign.design(
        spacing=(0.01, 0.2, 10), shift=(-4, 0, 10), save=False, **inp)

###############################################################################
# 2. First focus
# --------------

filt2, out2 = empymod.fdesign.design(
        spacing=(0.04, 0.1, 10), shift=(-3, -0.5, 10), save=False, **inp)

###############################################################################
# 3. Final focus
# --------------

filt, out = empymod.fdesign.design(
        spacing=(0.047, 0.08, 10), shift=(-2.4, -0.75, 10), finish=False,
        save=False, **inp)

###############################################################################
# To reproduce exactly the same filter as wer_201_2018:
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# ::
#
#     filt_orig, out_orig = fdesign.load_filter('wer201', True)
#     fdesign.plot_result(filt_orig, out_orig)
#     filt_orig, out_orig = fdesign.design(
#             spacing=out_orig[0][0], shift=out_orig[0][1], **inp)
#


###############################################################################
# Plot the result
# ---------------
#
# Plot function
# ~~~~~~~~~~~~~

def plotresult(depth, res, zsrc, zrec):
    x = np.arange(1, 101)*200
    inp = {
        'src': [0, 0, depth[1]-zsrc],
        'rec': [x, x*0, depth[1]-zrec],
        'depth': depth,
        'res': res,
        'ab': 11,
        'freqtime': 1,
        'verb': 1,
    }

    kong241 = empymod.dipole(htarg='kong_241_2007', **inp)
    key201 = empymod.dipole(htarg='key_201_2012', **inp)
    and801 = empymod.dipole(htarg='anderson_801_1982', **inp)
    test = empymod.dipole(htarg=filt, **inp)
    wer201 = empymod.dipole(htarg='wer_201_2018', **inp)
    qwe = empymod.dipole(ht='qwe', **inp)

    plt.figure(figsize=(8, 3.5))
    plt.subplot(121)
    plt.semilogy(x, qwe.amp, c='0.5', label='QWE')
    plt.semilogy(x, kong241.amp, 'k--', label='Kong241')
    plt.semilogy(x, key201.amp, 'k:', label='Key201')
    plt.semilogy(x, and801.amp, 'k-.', label='And801')
    plt.semilogy(x, test.amp, 'r', label='This filter')
    plt.semilogy(x, wer201.amp, 'b', label='Wer201')
    plt.legend()
    plt.xticks([0, 5e3, 10e3, 15e3, 20e3])
    plt.xlim([0, 20e3])

    plt.subplot(122)
    plt.semilogy(x, np.abs((kong241-qwe)/qwe), 'k--', label='Kong241')
    plt.semilogy(x, np.abs((key201-qwe)/qwe), 'k:', label='Key201')
    plt.semilogy(x, np.abs((and801-qwe)/qwe), 'k-.', label='And801')
    plt.semilogy(x, np.abs((test-qwe)/qwe), 'r', label='This filter')
    plt.semilogy(x, np.abs((wer201-qwe)/qwe), 'b', label='Wer201')
    plt.legend()
    plt.xticks([0, 5e3, 10e3, 15e3, 20e3])
    plt.xlim([0, 20e3])
    plt.ylim([1e-14, 1])

    plt.show()


###############################################################################
# Plot the individual models
# ~~~~~~~~~~~~~~~~~~~~~~~~~~

plotresult([-1e50, 2000], [2e14, 1/3.2, 1], 50, 0)  # KONG model
plotresult([0, 1000, 2000, 2100], [1/1e-12, 1/3.3, 1, 100, 1], 10, 0)  # KEY m.
plotresult([0, 1, 1000, 1100], [2e14, 10, 10, 500, 10], 0.5, 0.2)  # LAND model

###############################################################################

empymod.Report()
