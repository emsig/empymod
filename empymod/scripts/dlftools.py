r"""

``dlftools`` was developed for [GrKW21]_.

These routines can be useful to design digital linear filters for very
expensive kernels.

They need to be simplified and integrated properly into empymod (no need for an
extra module, :mod:`emg3d.scripts.fdesign` should be fine). For now, they are
added to the branch ``dlftools`` to not get forgotten or lost.

"""
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline as iuSpline


def get_dlf_values(time, filt, pts_per_dec=-1, omega_avail=None):
    r"""Return required angular frequencies.

    .. todo::

       Adjust and simplify the routine to use
       :func:`emg3d.utils.check_time`. Add tests and documentation.


    Returns the required angular frequencies for the given ``time``,
    ``filter``, and ``pts_per_dec``. If ``omega_avail`` is provided it uses
    :func:`expand_dlf_values`.

    The angular frequencies could be obtained via

       out = emg3d.utils.check_time(time, signal, ft, ftarg, verb)
       omegas = out[1]*2*np.pi

    """

    # Standard DLF
    if pts_per_dec == 0:
        omega = np.ravel(filt.base/time[:, None])
        dlf_vals = time

    else:
        # Get min and max required omega-values
        # (depends on filter-base and time)
        omega_max = filt.base[-1]/time.min()
        omega_min = filt.base[0]/time.max()

        if omega_avail is not None:

            # Ensure omega_avail is in omega; first minimum.
            if omega_min - omega_avail.min() < -omega_min*1e-6:
                print("* ERROR :: `time` needs frequencies below the ones "
                      "provided in `omega_avail`. *")
                print("  min(omega_avail) =", omega_avail.min())
                print("  min(omega)       =", omega_min)
                print("  Additional frequencies required:")
                _, add_below, _ = expand_dlf_values(
                        omega_avail, time, filt, pts_per_dec)
                print(add_below)
                raise ValueError('omega_avail')

            imin = np.where(omega_avail - omega_min <= omega_min*1e-6)[0]
            if len(imin) == 0:
                imin = 0
            else:
                imin = imin[-1]
            omega_min = omega_avail[imin]

    # Get nr_per_dec for lagged DLF.
    if pts_per_dec < 0:
        nr_per_dec = 1/np.log(filt.factor)

    # Calculate number of frequencies.
    if pts_per_dec < 0:    # Lagged DLF
        nout = int(np.ceil(np.log(omega_max/omega_min)*nr_per_dec) + 1)

        # Min-nout check, because the cubic InterpolatedUnivariateSpline needs
        # at least 4 points, and the lagged DLF interpolates in the output
        # domain.
        nout = max(nout, filt.base.size+3)

    elif pts_per_dec > 0:  # Splined DLF
        nout = int(np.ceil(np.log10(omega_max/omega_min)*pts_per_dec) + 1)

    # Calculate required frequencies.
    if pts_per_dec < 0:    # Lagged DLF
        omega = np.exp(
                np.arange(np.log(omega_min),
                          np.log(omega_min) + nout/nr_per_dec, 1/nr_per_dec))
    elif pts_per_dec > 0:  # Splined DLF
        omega = 10**np.arange(
                np.log10(omega_min), np.log10(omega_min) + nout/pts_per_dec,
                1/pts_per_dec)

    # Calculate the intermediate times (lagged) or intermediate frequencies
    # (splined) required for the DLF. Interpolation happens in the time-domain
    # (lagged) or frequency domain (splined), respectively. See notebook
    # 7a_DLF-Standard-Lagged-Splined.ipynb at
    # https://github.com/empymod/empymod-examples
    if omega_avail is not None:
        if pts_per_dec < 0:    # Lagged DLF
            dlf_vals = time.max()*np.exp(
                    -np.arange(nout - filt.base.size + 1) / nr_per_dec)
        elif pts_per_dec > 0:  # Splined DLF
            dlf_vals = filt.base/time[:, None]
        else:                  # Standard DLF
            dlf_vals = time

    # Return output values
    if omega_avail is not None:
        # Ensure omega_avail is in omega; second maximum.
        if omega.max() - omega_avail.max() > omega.max()*1e-6:
            print("* ERROR :: `time` needs frequencies above the ones "
                  "provided in `omega_avail`. *")
            print("  max(omega_avail) =", omega_avail.max())
            print("  max(omega)       =", omega.max())
            print("  Additional frequencies required:")
            _, _, add_above = expand_dlf_values(omega, time, filt, pts_per_dec)
            print(add_above)
            raise ValueError('omega_avail')

        return omega, dlf_vals
    else:
        return omega


def expand_dlf_values(omega, time, filt, pts_per_dec=-1):
    r"""Expand ``omega`` to suite all ``time``.

    .. todo::

       Add tests and document.

    """

    if pts_per_dec == 0:  # Standard DLF.
        print("Only implemented for lagged and splined DLF; returning.")
        return omega, [], []

    else:                 # Lagged or splined DLF.
        # Get min and max required omegas (depends on filter-base and time).
        act_omega = get_dlf_values(time, filt, pts_per_dec)
        omega_max = act_omega.max()
        omega_min = act_omega.min()

    # Check if we need to add frequencies.
    if omega.min() <= omega_min and omega.max() >= omega_max:
        print("No new frequencies required.")
        return omega, [], []

    # Initiate arrays.
    add_below = np.array([], dtype=float)
    add_above = np.array([], dtype=float)

    # Frequencies are in log or log10, depending if lagged or splined.
    if pts_per_dec < 0:    # Lagged DLF
        log = np.log
        exp = np.exp
    elif pts_per_dec > 0:  # Splined DLF

        def pow10(x):
            return 10**x

        log = np.log10
        exp = pow10

    # Get spacing.
    spacing = np.diff(log(omega))[0]

    # Add lower frequencies if required.
    while omega_min <= omega.min():
        new = exp(log(omega[0])-spacing)
        omega = np.r_[new, omega]
        add_below = np.r_[new, add_below]

    # Add higher frequencies if required.
    while omega_max >= omega.max():
        new = exp(log(omega[-1])+spacing)
        omega = np.r_[omega, new]
        add_above = np.r_[add_above, new]

    return omega, add_below, add_above


def dlf_sine(f_resp, omega, time, filt, pts_per_dec=-1):
    r"""Carry out DLF with a sine-filter.

    .. todo::

       Generalize to cosine as well. Add tests and document.

    A note re `time`:

    You can reuse already calculated `f_resp` used to calculated the response
    at times `time` to get the responses at other times `t_new`. Conditions
    are:

    - min(t_new) > min(time);
    - max(t_new) < max(time);
    - f_resp; omega; filt; dlf_vals; and pts_per_dec have not been changed.
    - `pts_per_dec` != 0 (only works for lagged and splined DLF).

    If these requirements are not met, the unexpected might happen. There is,
    as of now, not much checking implemented in this regard.

    """

    # Get omega for provided time (might be different from provided omega in
    # the lagged and splined DLF variants).
    req_omega, dlf_vals = get_dlf_values(time, filt, pts_per_dec, omega)

    # Restrict input f_resp and omega for wanted time.
    if pts_per_dec != 0:
        ii = np.searchsorted(omega, req_omega)
        f_resp = f_resp[ii]
        omega = omega[ii]

    # Re-arrange f-domain response.
    # Note: Sine transform uses negative imaginary part of response.
    if pts_per_dec < 0:
        tf_resp = np.concatenate(
                (np.tile(-f_resp.imag, dlf_vals.size).squeeze(),
                 np.zeros(dlf_vals.size)))
        tf_resp = tf_resp.reshape(dlf_vals.size, -1)[:, :filt.base.size]
    elif pts_per_dec > 0:
        tf_resp = iuSpline(np.log(omega), -f_resp.imag)(np.log(dlf_vals))
    else:
        tf_resp = -f_resp.imag.reshape(np.atleast_1d(time).size, -1)

    # Apply DLF
    t_resp = np.dot(tf_resp, filt.sin)

    # Lagged convolution DLF: interpolate to wanted times.
    if pts_per_dec < 0:
        t_resp = iuSpline(np.log(dlf_vals[::-1]), t_resp[::-1])(np.log(time))

    t_resp /= time

    return t_resp*2/np.pi  # Scaling from Fourier transform
