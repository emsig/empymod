"""
Transform utilities within empymod for other modellers
======================================================

This is an example how you can use the Fourier-transform tools implemented in
``empymod`` with other modellers. You could achieve the same for the Hankel
transform.

``empymod`` has various Fourier transforms implemented:

  - Digital Linear Filters DLF (Sine/Cosine)
  - Quadrature with Extrapolation QWE
  - Logarithmic Fast Fourier Transform FFTLog
  - Fast Fourier Transform FFT


For details of all the parameters see the ``empymod``-docs or the function's
docstrings.
"""
import empymod
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')

###############################################################################
# Model and transform parameters
# ------------------------------
#
# The model actually doesn't matter for our purpose, but we need some model to
# show how it works.

# Define model, a halfspace
model = {
    'src': [0, 0, 0.001],     # Source at origin, slightly below interface
    'rec': [6000, 0, 0.001],  # Receivers in-line, 0.5m below interface
    'depth': [0],             # Air interface
    'res': [2e14, 1],         # Resistivity: [air, half-space]
    'epermH': [0, 1],         # Set electric permittivity of air to 0 because
    'epermV': [0, 1],         # of numerical noise
}

# Specify desired times
time = np.linspace(0.1, 30, 301)

# Desired time-domain signal (0: impulse; 1: step-on; -1: step-off)
signal = 1

# Get required frequencies to model this time-domain result
# => we later need ``ft`` and ``ftarg`` for the Fourier transform.
# => See the docstrings (e.g., empymod.model.dipole) for available transforms
#    and their arguments.
time, freq, ft, ftarg = empymod.utils.check_time(
        time=time, signal=signal, ft='sin', ftarg={'pts_per_dec'}, verb=3)

###############################################################################
# Frequency-domain calculation
# ----------------------------
#
# **=> Here we calculate the frequency-domain result with `empymod`, but you
# could calculate it with any other modeller.**

fresp = empymod.dipole(freqtime=freq, **model)

###############################################################################
# Plot frequency-domain result

plt.figure()

plt.title('Frequency Domain')
plt.plot(freq, 1e9*fresp.real, 'C0.', label='Req. real frequencies')
plt.plot(freq, 1e9*fresp.imag, 'C1.', label='Req. imag frequencies')
plt.legend()
plt.xscale('log')
plt.xlabel('Frequency (Hz)')
plt.ylabel('$E_x$ (nV/m)')

plt.show()

###############################################################################
# Fourier transform
# -----------------

# Calculate corresponding time-domain signal.
tresp, _ = empymod.model.tem(
    fEM=fresp[:, None],
    off=model['rec'][0],
    freq=freq,
    time=time,
    signal=signal,
    ft=ft,
    ftarg=ftarg)

tresp = np.squeeze(tresp)

# Time-domain result just using empymod
tresp2 = empymod.dipole(freqtime=time, signal=signal, verb=1, **model)

###############################################################################
# Plot time-domain result

fig = plt.figure()

plt.title('Time domain')
plt.plot(time, tresp2*1e12, 'k', lw=2, label='empymod')
plt.plot(time, tresp*1e12, 'C0--', label='manual Fourier transform')
plt.legend()
plt.xlabel('Time (s)')
plt.ylabel('$E_x$ (uV/m)')

plt.show()

###############################################################################

empymod.Report()
