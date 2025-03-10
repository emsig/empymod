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

In this example we define a function to compute the in-phase and quadrature
components for a given model and a given system, and demonstrate it with the
GEM-2 and the DUALEM-842 equipments.


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
# In-phase and Quadrature function
# --------------------------------
#
# For a description of the parameters see :func:`empymod.model.dipole`.

def IPandQP(model, system, scale=1e3):
    """Return In-Phase and Quadrature components for provided model and system.

    This function takes two dictionaries, one collecting the model parameters,
    and one the system parameters, and a scaling factor. For a description of
    the parameters refer to the function `empymod.model.dipole`.

    Only implemented for magnetic sources and receivers, in the f-domain.


    Parameters
    ----------
    model : dict
        - Required: `depth`, `res`.
        - Optional: `aniso`, `epermH`, `epermV`, `mpermH`, `mpermV`

    system : dict
        - Required: `src`, `rec`, `freqtime`, and `ab` (only [4-6][4-6]).
        - Optional: `verb`, `ht`, `htarg`, `loop`, and `squeeze`.
        - Set: `xdirect` and `signal=None`; `ft`; `ftarg` have hence no effect.

    scale : float, default: 1e3 (ppt)
        Multiplication factor. E.g., 1e3 for ppt, 1e6 for ppm.


    Returns
    -------

    IP, QP : ndarrays
        In-phase and quadrature values.

    """

    # Ensure source and receiver are magnetic.
    if int(system.get('ab', 11)) not in [44, 45, 46, 54, 55, 56, 64, 65, 66]:
        raise ValueError("Only implemented for magnetic sources/receivers.")

    # Ensure signal is None.
    if system.get('signal'):
        raise ValueError("Only implemented for frequency domain.")

    # Secondary magnetic field (xdirect=None means no direct field)
    Hs = empymod.dipole(xdirect=None, **system, **model)

    # Primary magnetic field (a fullspace of air, hence ONLY the direct field)
    Hp = empymod.dipole(
        xdirect=True,
        # For PERP, ab=[46;64], Hp is zero; instead use Hp of the HCP config.
        # Frischknecht et al., 1991, p. 111; doi: 10.1190/1.9781560802686.ch3.
        **{
            **system,
            'ab': system['ab'] if system['ab'] not in [46, 64] else 66
        },
        # Take only the first value of each parameter, and set depth to empty.
        **{**{k: v[0] for k, v in model.items() if k != 'depth'}, 'depth': []}
    )

    # Take the ratio, multiply by scale
    H = scale * Hs / Hp

    # Return In-phase and Quadrature
    return H.real, H.imag


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

IP1, QP1 = IPandQP(model=model1, system=GEM2)

fig1, ax1 = plt.subplots(1, 1)
ax1.semilogx(freq, IP1, '-o', label='In-phase')
ax1.semilogx(freq, QP1, '-o', label='Quadrature')
ax1.set_title('GEM-2')
ax1.set_xlabel('Frequency (Hz)')
ax1.set_ylabel('IP or QP (ppt)')
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


def dualem(height, model, ab=[66, 55, 46], scale=1e3):
    """We wrap the system to accommodate the different configurations."""

    # The system
    DUALEM842S = {
        'src': [0, 0, -height],
        'freqtime': 9000,
        'verb': 1,
    }

    # Pre-allocate output
    IP = np.zeros((len(ab), 3))
    QP = np.zeros((len(ab), 3))

    # Loop over configurations
    for i, a in enumerate(ab):
        if a == 46:
            offset = [2.1, 4.1, 8.1]
        else:
            offset = [2.0, 4.0, 8.0]

        DUALEM842S['rec'] = [offset, [0, 0, 0], -height]
        DUALEM842S['ab'] = a

        IP[i, :], QP[i, :] = IPandQP(model, DUALEM842S, scale)

    return IP, QP


# Define your model
model2 = {
    'depth': [0, 2, 5],          # Half-space, 2m, 3m, half-space
    'res': [2e14, 50, 0.1, 50],  # Conductive middle layer
    'mpermH': [1, 1, 1.5, 1],    # Middle layer with a magn. susc. of 0.5
    # Optionally additional parameters: aniso, epermH, epermV, mpermV
}


ab = [66, 55, 46]
IP2, QP2 = dualem(height, model=model2, ab=ab)

fig2, ax2 = plt.subplots(1, 1)
off = [2, 4, 8]
ax2.plot(off, IP2.T, '-o', label=[f"IP ab={a}" for a in ab])
ax2.plot(off, QP2.T, '-o', label=[f"QP ab={a}" for a in ab])
ax2.set_title('DUALEM-842')
ax2.set_xlabel('Offset (m)')
ax2.set_ylabel('IP or QP (ppt)')
ax2.legend()


###############################################################################
empymod.Report()
