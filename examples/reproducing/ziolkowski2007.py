"""
Ziolkowski et al., 2007
=======================

Reproducing Figure 3 of Ziolkowski et al., 2007, Geophysics. This is a land
MTEM example.

**Reference**

- **Ziolkowski, A., B. Hobbs, and D. Wright,  2007**, Multitransient
  electromagnetic demonstration survey in France: Geophysics, 72, F197-F209;
  DOI: `10.1190/1.2735802 <https://doi.org/10.1190/1.2735802>`_.

"""
import empymod
import numpy as np
import matplotlib.pyplot as plt

###############################################################################
# Computation
# -----------

# Time
t = np.linspace(0.001, 0.06, 101)

# Target model
inp_tg = {
    'src': [0, 0, 0.001],
    'rec': [1000, 0, 0.001],
    'depth': [0, 500, 525],
    'res': [2e14, 20, 500, 20],
    'freqtime': t,
    'verb': 1,
}

# HS model
inp_hs = inp_tg.copy()
inp_hs['depth'] = inp_tg['depth'][0]
inp_hs['res'] = inp_tg['res'][:2]

# Compute responses
sths = empymod.dipole(**inp_hs, signal=1)               # Step, Halfspace
sttg = empymod.dipole(**inp_tg, signal=1)               # Step, Target
imhs = empymod.dipole(**inp_hs, signal=0, ft='fftlog')  # Impulse, Halfspace
imtg = empymod.dipole(**inp_tg, signal=0, ft='fftlog')  # Impulse, Target

###############################################################################
# Plot
# ----

fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(8, 4), sharex=True, constrained_layout=True)

# Step response
ax1.set_title('(a)')
ax1.plot(np.r_[0, 0, t], np.r_[0, sths[0], sths], 'k',
         label='Uniform half-space')
ax1.plot(np.r_[0, 0, t], np.r_[0, sttg[0], sttg], 'r',
         label='Hydrocarbon reservoir')
ax1.axis([-.02, 0.06, 0, 8e-9])
ax1.set_xlabel('Time (s)')
ax1.set_ylabel('Electric field amplitude (V/m/A-m)')
ax1.legend()

# Impulse response
ax2.set_title('(b)')
ax2.plot(np.r_[0, 0, t], np.r_[2, 0, imhs/imtg.max()],
         'k', label='Uniform half-space')
ax2.plot(np.r_[0, t], np.r_[0, imtg/imtg.max()],
         'r', label='Hydrocarbon reservoir')
ax2.axis([-.02, 0.06, 0, 1.02])
ax2.set_xlabel('Time (s)')
ax2.set_ylabel('Normalized derivative (Ω/m²/s)')
ax2.legend()

###############################################################################
# Original Figure
# ---------------
#
# Figure 3 of Ziolkowski et al., 2007, Geophysics:
#
# .. image:: ../../_static/figures/Ziolkowski2007.jpg
#

###############################################################################
empymod.Report()
