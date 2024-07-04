"""
Hunziker et al., 2015, Geophysics
=================================

Reproducing Figure 3 of the manual from ``EMmod``. This example does, as such,
not actually reproduce a figure of Hunziker et al., 2015, but of the manual
that comes with the software accompanying the paper. With the software comes an
example input file named ``simplemod.scr``, and the corresponding result is
shown in the manual of the code in Figure 3.

If you are interested in reproducing the figures of the actual paper, have a
look at the notebooks in the repo `article-geo2017
<https://github.com/emsig/article-geo2017>`_.

**Reference**

- **Hunziker, J., J. Thorbecke, and E. Slob, 2015**, The electromagnetic
  response in a layered vertical transverse isotropic medium: A new look at an
  old problem: Geophysics, 80(1), F1–F18; DOI: `10.1190/geo2013-0411.1
  <https://doi.org/10.1190/geo2013-0411.1>`_; Software:
  `wiki.seg.org/wiki/Software:emmod
  <https://wiki.seg.org/wiki/Software:emmod>`_.

"""
import empymod
import numpy as np
import matplotlib.pyplot as plt

###############################################################################
# Compute the data
# ----------------
#
# Compute the electric field with the parameters defined in ``simplemod.scr``.

# x- and y-offsets
x = np.arange(4000)*7 - 1999.5*7
y = np.arange(1500)*10 - 749.5*10

# Create 2D arrays of them
rx = np.tile(x[:, None], y.size).T
ry = np.tile(y[:, None], x.size)

# Compute the electric field
efield = empymod.dipole(
    src=[0, 0, -150],
    rec=[rx.ravel(), ry.ravel(), -200],
    depth=[0, -200, -1000, -1200],
    res=[2e14, 1/3, 1, 50, 1],
    aniso=[1, 1, np.sqrt(10), 1, 1],
    freqtime=0.5,
    epermH=[1, 80, 17, 2.1, 17],
    epermV=[1, 80, 17, 2.1, 17],
    mpermH=[1, 1, 1, 1, 1],
    mpermV=[1, 1, 1, 1, 1],
    ab=11,
    htarg={'pts_per_dec': -1},
).reshape(rx.shape)

###############################################################################
# Plot
# ----

# Create a similar colormap as Hunziker et al., 2015.
cmap = plt.get_cmap("jet", 61)

fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(8, 7), sharex=True, constrained_layout=True)

# 1. Amplitude
ax1.set_title('Amplitude (V/m)')
cf1 = ax1.pcolormesh(
        x/1e3, y/1e3, np.log10(efield.amp()),
        cmap=cmap, vmin=-16, vmax=-7, shading='nearest',
)
plt.colorbar(cf1, ticks=np.array([-16, -14, -12, -10, -8]))

# 2. Phase
ax2.set_title('Phase (°)')
ax2.set_xlabel('Offset (km)')
cf2 = ax2.pcolormesh(
        x/1e3, y/1e3, efield.pha(deg=False, unwrap=False, lag=True),
        cmap=cmap, vmin=-np.pi, vmax=np.pi, shading='nearest',
)
plt.colorbar(cf2, ticks=np.array([-2, 0, 2]))

for ax in [ax1, ax2]:
    ax.axis('equal')
    ax.set_ylim([y.max(), y.min()])
    ax.set_yticks([5, 0, -5])
    ax.set_ylabel('Offset (km)')


###############################################################################
# Original Figure
# ---------------
#
# Figure 3 of the manual of `EMmod`.
#
# .. image:: ../../_static/figures/Hunziker2015.png

###############################################################################
empymod.Report()
