import numpy as np
import ctypes
import multiprocessing


# TODO: Wrap methods of ``Frame`` here
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


# It is a one step for next function
def de_disperse_freq_average(params):
    """
    De-disperse frame using specified value of DM and average in frequency.
    :param dm:
        Dispersion measure to use in de-dispersion [cm^3 / pc].
    :notes:
        This method avoids creating ``(n_nu, n_t)`` arrays and must be
        faster for data with big sizes. But it returns already frequency
        averaged de-dispersed dyn. spectra.
    """
    dm, dsp, nu, nu_max, d_t = params
    n_nu, n_t = dsp.shape

    # Calculate shift of time caused by de-dispersion for all channels
    dt_all = k * dm * (1. / nu ** 2. - 1. / nu_max ** 2.)
    # Find what number of time bins corresponds to this shifts
    nt_all = vint(vround(dt_all / d_t))
    # Container for summing de-dispersed frequency channels
    values = np.zeros(n_t)
    # Roll each axis (freq. channel) to each own number of time steps.
    for i in range(n_nu):
        values += np.roll(dsp[i], -nt_all[i])

    return values / n_nu


def noncoherent_dedisperse(dsp, dm_grid, threads=1, **kwargs):
    """
    Method that de-disperse dynamical spectra with range values of dispersion
    measures and average them in frequency to obtain image in (t, DM)-plane.

    :param dsp:
        Dynamical spectra numpy array.
    :param dm_grid:
        Array-like of value of DM on which to de-disperse [cm^3/pc].
    :param threads: (optional)
        Number of threads used for parallelization with ``multiprocessing``
        module. If ``1`` then it isn't used. (default: 1)
    :param kwargs:
        Keyword arguments that should contain ``nu_max``, ``d_nu`` & ``d_t``
        parameters. See code for explanation.
    """
    # Frequency of highest frequency channel [MHz].
    nu_max = kwargs['nu_max']
    # Width of spectral channel [MHz].
    d_nu = kwargs['d_nu']
    # Time step [s].
    d_t = kwargs['d_t']

    n_nu, n_t = dsp.shape
    nu = np.arange(n_nu, dtype=float)
    nu = (nu_max - nu * d_nu)[::-1]

    pool = None
    if threads > 1:
        pool = multiprocessing.Pool(threads, maxtasksperchild=1000)

    if pool:
        m = pool.map
    else:
        m = map

    dsp_shared = to_shared_array(dsp)

    params = [(dm, dsp_shared, nu, nu_max, d_t) for dm in dm_grid]

    # Accumulator of de-dispersed frequency averaged frames
    frames = list(m(de_disperse_freq_average, params))
    frames = np.array(frames)

    if pool:
        # Close pool
        pool.close()
        pool.join()

    return frames


def to_shared_array(array):
    """
    Function that creates shared array with data - copy of data in user supplied
    array.
    :param array:
        2D numpy.ndarray from which to make shared array.

    :return:
        Shared array.
    """
    # Using shared array (http://stackoverflow.com/questions/5549190 by pv.)
    i, j = array.shape
    shared_array_base = multiprocessing.Array(ctypes.c_float, i * j)
    shared_array =  np.ctypeslib.as_array(shared_array_base.get_obj()).reshape((i, j))
    shared_array += array.copy()
    return shared_array
