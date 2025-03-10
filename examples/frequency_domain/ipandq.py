"""
In-phase and Quadrature
=======================

Profiling measurements using small sources are often analysed in terms of
in-phase and quadrature readings, see, e.g., Frischknecht et al., 1991. In this
method, the in-phase and quadrature components are the real and imaginary part,
respectively, of the ratio of the secondary field (total field without direct
wave) over the primary field (direct wave, usually the airwave). This ratio is
often multiplied by a factor, and then given as, for instance, ppt (factor 1e3)
or ppm (factor 1e6).

This example demonstrates the use of :func:`empymod.model.ip_and_q`, the
wrapper to obtain in-phase and quadrature components for a given model and a
given system, with the GEM-2 and the DUALEM-842 equipments.


**Reference**

- **Frischknecht, F.C., V.F. Labson, B.R. Spies, and W.L. Anderson, 1991**,
  Profiling Methods Using Small Sources: Chapter 3 in *Electromagnetic Methods
  in Applied Geophysics: Volume 2, Application, Parts A and B*, 105-270; DOI:
  `10.1190/1.9781560802686.ch3 <https://doi.org/10.1190/1.9781560802686.ch3>`_.


With the help and expertise of María Carrizo Mascarell (`@mariacarrizo
<https://github.com/mariacarrizo>`_).

"""
import empymod
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')


###############################################################################
# GEM-2
# -----
#
# The GEM-2 is a fixed beam system with a transmitter coil and a receiver coil,
# manufactured by `Geophex Ltd. <https://geophex.com>`_. It can operate up to
# 10 frequencies btw. 30 Hz and 93 kHz.

# Define your configuration
height = 1.0  # Your equipment height above ground
ab = 66       # Your configuration: 66 for HCP or 55 for VCP
# Your chosen frequencies
freq = np.logspace(np.log10(30), np.log10(93000), 10)

# Collect system details
GEM2 = {
    'src': [0, 0, -height],
    'rec': [1.66, 0, -height],  # GEM-2 has a coil separation of 1.66 m
    'freqtime': freq,
    'ab': ab,
    'verb': 1,
}

# Define your model
model1 = {
    'depth': [0, 2, 5],          # Half-space, 2m, 3m, half-space
    'res': [2e14, 50, 0.1, 50],  # Conductive middle layer
    'mpermH': [1, 1, 1.5, 1],    # Middle layer with a magn. susc. of 0.5
    # Optionally additional parameters: aniso, epermH, epermV, mpermV
}

IP1, Q1 = empymod.ip_and_q(**model1, **GEM2)

fig1, ax1 = plt.subplots(1, 1)
ax1.semilogx(freq, IP1, '-o', label='In-phase')
ax1.semilogx(freq, Q1, '-o', label='Quadrature')
ax1.set_title('GEM-2')
ax1.set_xlabel('Frequency (Hz)')
ax1.set_ylabel('IP or Q (ppt)')
ax1.legend()


###############################################################################
# DUALEM-842
# ----------
#
# The DUALEM-842 is a fixed beam system with a transmitter coil and six
# receiver coil: one coplanar at 2, 4, and 8 m offset, and one perpendicular at
# 21., 4.1, and 8.1 m offset. Operation frequency is 9 kHz. It is manufactured
# by `Dualem Inc. <https://dualem.com>`_.

height = 1.0  # Equipment height


def dualem(height, model, ab, scale=1e3):
    """We wrap the system to accommodate the different configurations."""

    # The system
    DUALEM842S = {
        'src': [0, 0, -height],
        'freqtime': 9000,
        'verb': 1,
    }

    # Pre-allocate output
    IP = np.zeros((len(ab), 3))
    Q = np.zeros((len(ab), 3))

    # Loop over configurations
    for i, a in enumerate(ab):
        if a == 46:
            offset = [2.1, 4.1, 8.1]
        else:
            offset = [2.0, 4.0, 8.0]

        DUALEM842S['rec'] = [offset, [0, 0, 0], -height]
        DUALEM842S['ab'] = a

        IP[i, :], Q[i, :] = empymod.ip_and_q(
            **model, **DUALEM842S, scale=scale
        )

    return IP, Q


# Define your model
model2 = {
    'depth': [0, 2, 5],          # Half-space, 2m, 3m, half-space
    'res': [2e14, 50, 0.1, 50],  # Conductive middle layer
    'mpermH': [1, 1, 1.5, 1],    # Middle layer with a magn. susc. of 0.5
    # Optionally additional parameters: aniso, epermH, epermV, mpermV
}


ab = [66, 55, 46]
IP2, Q2 = dualem(height, model=model2, ab=ab)

fig2, ax2 = plt.subplots(1, 1)
off = [2, 4, 8]
ax2.plot(off, IP2.T, '-o', label=[f"IP ab={a}" for a in ab])
ax2.plot(off, Q2.T, '-o', label=[f"Q ab={a}" for a in ab])
ax2.set_title('DUALEM-842')
ax2.set_xlabel('Offset (m)')
ax2.set_ylabel('IP or Q (ppt)')
ax2.legend()


###############################################################################
empymod.Report()
