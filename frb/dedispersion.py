import numpy as np


vint = np.vectorize(int)
vround = np.vectorize(round)
# MHz ** 2 * cm ** 3 * s / pc
k = 1. / (2.410331 * 10 ** (-4))


def de_disperse(dyn_spectr, nu_0, d_nu, d_t, dm_values):
    """
    De-disperse dynamical spectra with grid of user specifies values of DM.

    :param dyn_spectr:
        2D numpy array of dynamical spectra (#freq, #t).
    :param nu_0:
        Frequency of highest frequency channel [MHz].
    :param d_nu:
        Width of spectral channel [MHz].
    :param d_t:
        Time step [s].
    :param dm_values:
        Array-like of DM values to de-disperse [cm^3 /pc].

    :return:
        2D numpy array (a.k.a. TDM-array) (#DM, #t)

    :notes:
        Probably, it won't work (at least efficiently) when time shift between
        close frequency channels > one time interval.
    """
    dm_values = np.array(dm_values)
    n_nu, n_t = dyn_spectr.shape
    nu = np.arange(n_nu, dtype=float)
    nu = (nu_0 - nu * d_nu)[::-1]
    # Pre-calculating cumulative sums and their difference
    cumsums = np.cumsum(dyn_spectr[::-1, :], axis=0)
    dcumsums = np.roll(cumsums, 1, axis=1) - cumsums

    # Calculate shift of time caused by de-dispersion for all channels and all
    # values of DM
    dt_all = k * dm_values[:, np.newaxis] * (1. / nu ** 2. - 1. / nu_0 ** 2.)
    # Find what number of time bins corresponds to this shifts
    nt_all = vint(vround(dt_all / d_t))[:, ::-1]

    # Create array for TDM
    values = np.zeros((len(dm_values), n_t), dtype=float)
    # Fill DM=0 row
    values[0] = cumsums[-1]

    # Cycle over DM values and fill TDM array for others DM values
    for i, nt in enumerate(nt_all[1:]):
        # Find at which frequency channels time shifts have occurred
        indx = np.array(np.where(nt[1:] - nt[:-1] == 1)[0].tolist() +
                        [n_nu - 1])
        result = np.roll(cumsums[-1], -nt[-1])
        for ix, j in enumerate(indx[:-1]):
            result += np.roll(dcumsums[j], -nt[j])
        values[i + 1] = result

    return values
