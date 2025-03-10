"""
In-phase and Quadrature
=======================



With the help and contributions of María Carrizo Mascarell (`@mariacarrizo
<https://github.com/mariacarrizo>`_).

"""
import empymod
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')


###############################################################################
# In-phase and Quadrature function
# --------------------------------

def IPandQP(model, system, scale=1e6):
    """Return In-Phase and Quadrature for provided model and system.

    Uses the function `empymod.dipole` with `xdirect=None` for the secondary
    field, and with `xdirect=True` for the analytical fullspace solution for
    the primary field.

    Parameters
    ----------
    model : dict
        Requires `depth`, `res`,
        Optional are `aniso`, `epermH`, `epermV`, `mpermH`, `mpermV`

    system : dict
        Requires `src`, `rec`, `freqtime`, and `ab`.
        Optional are `verb`, `ht`, and `htarg`, `loop`, and `squeeze`.
        Set are `signal=None`; `ft` and `ftarg` have hence no effect.

    scale : float, default: 1e6 (ppm)
        Scale with which the ratio is multiplied. E.g., 1e3 for ppt, 1e6 for
        ppm.


    Returns
    -------

    IP, QP : ndarrays
        In-phase and quadrature values.
    """

    # Secondary magnetic field (xdirect=None means no direct field / no
    # airwave)
    Hs = empymod.dipole(xdirect=None, **system, **model)

    # Primary magnetic field (a fullspace of air, hence ONLY the direct field /
    # primary field)
    Hp = empymod.dipole(
        xdirect=True,
        # For PERP (46, 64), the Hp is zero; the HCP-Hp is used instead.
        # Frischknecht et al. 1991, p. 111 (EM Methods in Applied Geophysics,
        # Vol 2)
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
# https://geophex.com

height = 1.0  # Equipment height
ab = 66.      # 66 for HCP or 55 for VCP

GEM2 = {
    'src': [0, 0, -height],
    'rec': [1.66, 0, -height],  # coil separation of 1.66 m
    'freqtime': np.array([475, 1625, 5475, 18575, 63025]),
    'ab': ab,
    'verb': 1,
}


model = {
    'depth': [0, 2, 5],
    'res': [2e14, 50, 0.1, 50],
    # Other parameters
}

IP, QP = IPandQP(model=model, system=GEM2)


###############################################################################
# DUALEM-842
# ----------
#
# https://dualem.com

height = 1.0  # Equipment height


def dualem(height, model, ab=[66, 55, 46], scale=1e3):
    DUALEM842S = {
        'src': [0, 0, -height],
        'freqtime': 9000,
        'verb': 1,
    }

    IP = np.zeros((len(ab), 3))
    QP = np.zeros((len(ab), 3))

    for i, a in enumerate(ab):
        if a == 46:
            offset = [2.1, 4.1, 8.1]
        else:
            offset = [2.0, 4.0, 8.0]

        DUALEM842S['rec'] = [offset, [0, 0, 0], -height]
        DUALEM842S['ab'] = a

        IP[i, :], QP[i, :] = IPandQP(model, DUALEM842S, scale)

    return IP, QP


model = {
    'depth': [0, 2, 5],
    'res': [2e14, 50, 0.1, 50],
    # Other parameters
}

IP, QP = dualem(height, model=model)


###############################################################################
empymod.Report()
