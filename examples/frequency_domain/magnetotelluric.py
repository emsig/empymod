"""
Magnetotelluric data
====================

"""
import empymod
import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import mu_0
plt.style.use('ggplot')

###############################################################################
# Define model parameter and frequencies
# --------------------------------------

resistivities = np.array([2e14, 300, 2500, 0.8, 3000, 2500])
depths = np.array([0, 200, 600, 640, 1140])
frequencies = np.logspace(-4, 4, 21)

###############################################################################
# 1D-MT
# ~~~~~
#
# Strongly modified after the example code of Andrew Pethick:
# https://www.digitalearthlab.com/tutorial/tutorial-1d-mt-forward/ [(c) WTFPL]

# Step 1. Compute basement impedance
impedance = np.sqrt(2j * np.pi * frequencies * mu_0 * resistivities[-1])

# Step 2. Iterate from second-last to top layer (without air)
for j in range(resistivities.size-2, 0, -1):

    # Step 2.1 Calculate intrinsic impedance of current layer
    dj = np.sqrt(2j * np.pi * frequencies * mu_0 / resistivities[j])
    wj = dj * resistivities[j]

    # Step 2.2 Calculate exponential factor from intrinsic impedance
    ej = np.exp(-2 * (depths[j]-depths[j-1]) * dj)

    # Step 2.3 Calculate reflection coeficient using current layer
    #          intrinsic impedance and the previous layer impedance
    re = ej * (wj - impedance) / (wj + impedance)
    impedance = wj * ((1 - re)/(1 + re))

# Step 3. Compute apparent resistivity last impedance
ares_mt1d = abs(impedance)**2/(2 * np.pi * frequencies * mu_0)
pha_mt1d = np.arctan2(impedance.imag, impedance.real)


###############################################################################
# Reproducing using empymod
# -------------------------
#
# The above 1D MT code assumes plane waves. We can "simulate" plane waves by
# putting the source far away. In this example, we set it one million km away
# in all directions. As the impedance is the ratio of the Ex and Hy fields the
# source type (electric or magnetic) and the source orientation do not matter.

dist = 1_000_000_000  # 1 million kilometer (!)
inp = {
    'src': (-dist, -dist, -dist),
    'rec': (0, 0, 0.1),
    'res': resistivities,
    'depth': depths,
    'freqtime': frequencies,
    'verb': 1,
}

ex = empymod.dipole(ab=11, **inp)
hy = empymod.dipole(ab=51, **inp)
Z = ex/hy
ares_empy = abs(Z)**2 / (2*np.pi*frequencies*mu_0)
pha_empy = np.arctan2(Z.imag, Z.real)


###############################################################################
# Plot results
# ------------
#
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4))

ax1.set_title('Apparent resistivity')
ax1.plot(frequencies, ares_mt1d, label='MT-1D')
ax1.plot(frequencies, ares_empy, 'o', label='empymod')
ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.set_xlabel('Frequency (Hz)')
ax1.set_ylabel(r'Apparent resistivity ($\Omega\,$m)')
ax1.legend()

ax2.set_title('Phase')
ax2.plot(frequencies, pha_mt1d*180/np.pi)
ax2.plot(frequencies, pha_empy*180/np.pi, 'o')
ax2.set_xscale('log')
ax2.yaxis.tick_right()
ax2.set_xlabel('Frequency (Hz)')
ax2.set_ylabel('Phase (degree)')
ax2.yaxis.set_label_position("right")

fig.tight_layout()
fig.show()


###############################################################################
empymod.Report()
