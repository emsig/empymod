"""Generate the IP/Q reference data in ppt with SimPEG 0.26.0.

Run ``python -I tests/create_data/simpeg.py`` from the repository root.
Isolated mode lets the script import the installed SimPEG package.
"""
from pathlib import Path

import numpy as np
from simpeg import maps
from simpeg.electromagnetics import frequency_domain as fdem


def reference(frequency):
    # SimPEG uses z up. Both coils are 0.16 m above the earth, 10 m apart.
    location = [6., 8., 0.16]
    receivers = [
        fdem.receivers.PointMagneticFieldSecondary(
            locations=location, orientation=axis, component=component,
            data_type="field")
        for axis in "xyz" for component in ("real", "imag")
    ]
    source = fdem.sources.MagDipole(
        receivers, frequency=frequency, location=[0., 0., 0.16],
        orientation="z", moment=1.)
    simulation = fdem.Simulation1DLayered(
        survey=fdem.Survey([source]), thicknesses=[1., 1.],
        sigmaMap=maps.IdentityMap(nP=3))
    secondary = simulation.dpred([0.001, 0.1, 0.001]).reshape(3, 2)

    # Reversing the z axis changes the sign of the two cross-components.
    secondary[:2] *= -1
    # Vertical source and coplanar reference receiver, unit dipole moment.
    primary = -1 / (4 * np.pi * 10**3)
    return 1e3 * secondary / primary


if __name__ == "__main__":
    frequencies = np.array([1000., 10000.])
    readings = np.stack([reference(f) for f in frequencies])
    output = Path(__file__).resolve().parents[1] / "data" / "simpeg.npz"
    np.savez_compressed(
        output, frequencies=frequencies, **{
            str(ab): readings[:, i, :] for i, ab in enumerate((46, 56, 66))
        })
