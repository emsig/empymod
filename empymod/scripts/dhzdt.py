import numpy as np
from scipy.constants import mu_0
from scipy.special import erf

def dhzdt(t, sigma, r):
    r"""Return the impulse response (i.e., dHz(t)/dt) of a homogeneous isotropic half-space
    of electrical conductivity sigma in z>=0 due to a current shut-off in a vertical
    magnetic dipole source of unit moment for times t>0.

    Source and receiver are both at the plane z=0 and separated by the distance r.
    """
    theta = np.sqrt(mu_0 * sigma / 4 / t)
    theta_r = theta * r
    s = 9 * erf(theta_r)
    s -= 2 * theta_r /np.sqrt(np.pi) * (9 + 6 * theta_r**2 + 4 * theta_r**4) * np.exp(-theta_r**2)
    s /= (-2 * np.pi * mu_0 * sigma * r**5)
    return s