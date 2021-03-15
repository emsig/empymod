r"""
Magnetotelluric data
====================

As background theory we reproduce here Equations (11) to (17) from Pedersen and
Hermance (1986), with surrounding text:

--------

If we define the impedance as

.. math::
    :label: ph-11

    Z = E_x / H_y \, ,

[...]  we can develop a recursive relationship for the impedance at the top of
the j-th layer looking down

.. math::
    :label: ph-12

    Z_j = z_{oj} \frac{1-R_j \exp(-2\gamma_j t_j)}{1+R_j \exp(-2\gamma_j t_j)}
        \, , \quad
    j = N-1, \dots, 1 \, ,

where

.. math::
    :label: ph-13

    z_{oj} \equiv \text{intrinsic impedance}
        \equiv \sqrt{\rm{i}\omega \mu \rho_j} \, ,

.. math::
    :label: ph-14

    R_j \equiv \text{reflection coefficient}
        \equiv \frac{z_{oj} - Z_{j+1}}{z_{oj} + Z_{j+1}} \, ,

.. math::

    t_j = \text{thickness of layer} j \, ,

and the impedance at the surface of the deepest layer (:math:`j=N`) is given by

.. math::
    :label: ph-15

    Z_N = z_{oN}\, .

The surface impedance, :math:`Z_j`, is found by applying Equation :eq:`ph-12`
recursively from the top of the bottom half-space, :math:`j = N`, and
propagating upwards. From the surface impedance, :math:`Z_1`, we can then
calculate the apparent resistivity, :math:`\rho_a`, and phase,
:math:`\theta_a`, as

.. math::
    :label: ph-16

    \rho_a = \frac{|Z_1|^2}{\omega \mu} \, ,

.. math::
    :label: ph-17

    \theta_a = \tan^{-1}\frac{\Im(Z1)}{\Re(Z_1)} \, ,

This calculation is repeated for a range of periods and is used to model the
magnetotelluric response of the layered structure.

--------

**Reference**:

- Pedersen, J., Hermance, J.F. Least squares inversion of one-dimensional
  magnetotelluric data: An assessment of procedures employed by Brown
  University. Surv Geophys 8, 187–231 (1986).
  https://doi.org/10.1007/BF01902413
- This example was strongly motivated by Andrew Pethicks blog post
  https://www.digitalearthlab.com/tutorial/tutorial-1d-mt-forward.



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
frequencies = np.logspace(-4, 4, 31)
omega = 2 * np.pi * frequencies

###############################################################################
# 1D-MT following Pedersen & Hermance
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Initiate recursive formula with impedance of the deepest layer.
impedance = np.sqrt(2j * np.pi * frequencies * mu_0 * resistivities[-1])

# Move up the stack of layers till the top (without air).
for l in range(depths.size-1, 0, -1):

    # Thickness
    t_j = depths[l] - depths[l-1]

    # Intrinsic impedance
    z_oj = np.sqrt(1j * omega * mu_0 * resistivities[l])

    # Reflection coefficient
    refl_coeff = (z_oj - impedance) / (z_oj + impedance)

    # Exponential factor
    gamma = np.sqrt(1j * omega * mu_0 / resistivities[l])
    exp_fact = np.exp(-2 * gamma * t_j)

    # Impedance at this layer
    impedance = z_oj * (1 - refl_coeff*exp_fact) / (1 + refl_coeff*exp_fact)

# Step 3. Compute apparent resistivity last impedance
apres_mt1d = abs(impedance)**2/(omega * mu_0)
phase_mt1d = np.arctan2(impedance.imag, impedance.real)

###############################################################################
# Reproducing using empymod
# -------------------------
#
# The above derivation and code assume plane waves. We can "simulate" plane
# waves by putting the source far away. In this example, we set it one million
# km away in all directions. As the impedance is the ratio of the Ex and Hy
# fields the source type (electric or magnetic) and the source orientation do
# not matter.

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
apres_empy = abs(Z)**2 / (2*np.pi*frequencies*mu_0)
phase_empy = np.arctan2(Z.imag, Z.real)

###############################################################################
# Plot results
# ------------
#
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4))

ax1.set_title('Apparent resistivity')
ax1.plot(frequencies, apres_mt1d, label='MT-1D')
ax1.plot(frequencies, apres_empy, 'o', label='empymod')
ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.set_xlabel('Frequency (Hz)')
ax1.set_ylabel(r'Apparent resistivity ($\Omega\,$m)')
ax1.legend()

ax2.set_title('Phase')
ax2.plot(frequencies, phase_mt1d*180/np.pi)
ax2.plot(frequencies, phase_empy*180/np.pi, 'o')
ax2.set_xscale('log')
ax2.yaxis.tick_right()
ax2.set_xlabel('Frequency (Hz)')
ax2.set_ylabel('Phase (degree)')
ax2.yaxis.set_label_position("right")

fig.tight_layout()
fig.show()


###############################################################################
empymod.Report()
