"""
Constable and Weiss, 2006
=========================

Reproducing Figure 3 of Constable and Weiss, 2006, Geophysics. This is a marine
CSEM example.

**Reference**

- **Constable, S., and C. J. Weiss,  2006**, Mapping thin resistors and
  hydrocarbons with marine EM methods: Insights from 1D modeling: Geophysics,
  71, G43-G51; DOI: `10.1190/1.2187748 <https://doi.org/10.1190/1.2187748>`_.

"""
import empymod
import numpy as np
import matplotlib.pyplot as plt

empymod.set_minimum(min_off=1e-10)

###############################################################################
# Computation
# -----------
#
# Note: Exact reproduction is not possible, as source and receiver depths are
# not explicitly specified in the publication. I made a few checks, and it
# looks like a source-depth of 900 meter gives good accordance. Receivers are
# on the sea-floor.

# Offsets
x = np.linspace(0, 20000, 101)

# TG model
inp_tg = {
    'src': [0, 0, 900],
    'rec': [x, 0, 1000],
    'depth': [0, 1000, 2000, 2100],
    'res': [2e14, 0.3, 1, 100, 1],
    'freqtime': 1,
    'verb': 1,
}

# HS model
inp_hs = inp_tg.copy()
inp_hs['depth'] = inp_tg['depth'][:2]
inp_hs['res'] = inp_tg['res'][:3]

# Compute radial responses
rhs = empymod.dipole(ab=11, **inp_hs)  # Halfspace
rtg = empymod.dipole(ab=11, **inp_tg)  # Target

# Compute azimuthal response
ahs = empymod.dipole(ab=22, **inp_hs)  # Halfspace
atg = empymod.dipole(ab=22, **inp_tg)  # Target


###############################################################################
# Plot
# ----

fig, axs = plt.subplots(3, 2, figsize=(9, 13), constrained_layout=True)
oldsettings = np.geterr()
_ = np.seterr(all='ignore')

# Radial amplitude
axs[0, 0].set_title('(a) Radial mode fields')
axs[0, 0].plot(x/1000, np.log10(rtg.amp()), 'k', label='Model')
axs[0, 0].plot(x/1000, np.log10(rhs.amp()), 'k-.', label='Half-space response')
axs[0, 0].axis([0, 20, -18, -8])
axs[0, 0].set_xlabel('Range (km)')
axs[0, 0].set_xticks([0, 5, 10, 15, 20])
axs[0, 0].set_ylabel('Log10(E-field magnitude, V/Am²)')
axs[0, 0].legend()

# Radial phase
axs[1, 0].set_title('(b) Radial mode phase')
axs[1, 0].plot(x/1000, rtg.pha(deg=True), 'k')
axs[1, 0].plot(x/1000, rhs.pha(deg=True), 'k-.')
axs[1, 0].axis([0, 20, -500, 0])
axs[1, 0].set_xlabel('Range (km)')
axs[1, 0].set_xticks([0, 5, 10, 15, 20])
axs[1, 0].set_ylabel('Phase (degrees)')

# Azimuthal amplitude
axs[2, 0].set_title('(c) Azimuthal mode fields')
axs[2, 0].plot(x/1000, np.log10(atg.amp()), 'k', label='Model')
axs[2, 0].plot(x/1000, np.log10(ahs.amp()), 'k-.', label='Half-space response')
axs[2, 0].axis([0, 20, -18, -8])
axs[2, 0].set_xlabel('Range (km)')
axs[2, 0].set_xticks([0, 5, 10, 15, 20])
axs[2, 0].set_ylabel('Log10(E-field magnitude, V/Am²)')
axs[2, 0].legend()

# Azimuthal phase
axs[0, 1].set_title('(d) Azimuthal mode phase')
axs[0, 1].plot(x/1000, atg.pha(deg=True)+180, 'k')
axs[0, 1].plot(x/1000, ahs.pha(deg=True)+180, 'k-.')
axs[0, 1].axis([0, 20, -500, 0])
axs[0, 1].set_xlabel('Range (km)')
axs[0, 1].set_xticks([0, 5, 10, 15, 20])
axs[0, 1].set_ylabel('Phase (degrees)')

# Normalized
axs[1, 1].set_title('(e) Normalized E-field magnitude')
axs[1, 1].plot(x/1000, np.abs(rtg/rhs), 'k', label='Radial')
axs[1, 1].plot(x/1000, np.abs(atg/ahs), 'k--', label='Azimuthal')
axs[1, 1].axis([0, 20, 0, 70])
axs[1, 1].set_xlabel('Range (km)')
axs[1, 1].set_xticks([0, 5, 10, 15, 20])
axs[1, 1].legend()

axs[2, 1].axis('off')

_ = np.seterr(**oldsettings)

###############################################################################
# Original Figure
# ---------------
#
# Figure 3 of Constable and Weiss, 2006, Geophysics:
#
# .. image:: ../../_static/figures/Constable2006.jpg
#

###############################################################################
empymod.Report()
