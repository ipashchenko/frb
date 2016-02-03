import numpy as np


vint = np.vectorize(int)
vround = np.vectorize(round)
# MHz ** 2 * cm ** 3 * s / pc
k = 1. / (2.410331 * 10 ** (-4))


def de_disperse(dyn_spectr, dm_values, *args, **kwargs):
    """
    De-disperse dynamical spectra with grid of user specifies values of DM.

    :param dyn_spectr:
        2D numpy array of dynamical spectra (#freq, #t).
    :param dm_values:
        Array-like of DM values to de-disperse [cm^3 /pc].
    :param kwargs:
        Keyword arguments that should contain ``nu_max``, ``d_nu`` & ``d_t``
        parameters. See code for explanation.

    :return:
        2D numpy array (a.k.a. TDM-array) (#DM, #t)

    :notes:
        Probably, it won't work (at least efficiently) when time shift between
        close frequency channels > one time interval.
    """
    # Frequency of highest frequency channel [MHz].
    nu_max = kwargs['nu_max']
    # Width of spectral channel [MHz].
    d_nu = kwargs['d_nu']
    # Time step [s].
    d_t = kwargs['d_t']
    dm_values = np.array(dm_values)
    n_nu, n_t = dyn_spectr.shape
    nu = np.arange(n_nu, dtype=float)
    # FIXME: I calculate it when reading FITS
    nu = (nu_max - nu * d_nu)[::-1]
    # Pre-calculating cumulative sums and their difference
    cumsums = np.cumsum(dyn_spectr[::-1, :], axis=0)
    dcumsums = np.roll(cumsums, 1, axis=1) - cumsums

    # Calculate shift of time caused by de-dispersion for all channels and all
    # values of DM
    dt_all = k * dm_values[:, np.newaxis] * (1. / nu ** 2. - 1. / nu_max ** 2.)
    # Find what number of time bins corresponds to this shifts
    nt_all = vint(vround(dt_all / d_t))[:, ::-1]

    # Create array for TDM
    values = np.zeros((len(dm_values), n_t), dtype=float)
    # FIXME: Generally there could be nonzero list of DM values
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


def combain_tdm(tdm1, tdm2, nu_high1, d_nu1, nu_high2, d_t, dm_values):
    """
    Function that adds t-DM arrays from dedispersion of different frequency
    bands.

    :param tdm1:
        2D array of de-dispersed values of first (high frequency) band.
    :param tdm2:
        2D array of de-dispersed values of second (low frequency) band.
    :param nu_high1:
        Highest frequency of first band.
    :param d_nu1:
        Frequency width of first band.
    :param nu_high2:
        Highest frequency of second band.
    :param d_t:
        Time step [s].
    :param dm_values:
        Array-like of DM values to de-disperse [cm^3 /pc].

    :return:
        2D TDM-array of de-dispersed values combined.
    """
    assert np.shape(tdm1) == np.shape(tdm2)
    assert np.shape(tdm1)[0] == len(dm_values)

    # Calculate shift of time caused by de-dispersion in first (high frequency)
    # band for all values of DM
    dt_all = k * dm_values * (1. / (nu_high1 + d_nu1) ** 2. -
                              1. / nu_high1 ** 2.)
    # Find what number of time bins corresponds to this shifts
    nt_all = vint(vround(dt_all / d_t))

    # Create array for TDM
    values = np.zeros(np.shape(tdm1), dtype=float)

    # Cycle over DM values and fill TDM array for others DM values
    for i, nt in enumerate(nt_all):
        # Find at which frequency channels time shifts have occurred
        values[i] = tdm1[i] + np.roll(tdm2[i], -nt)

    return values
