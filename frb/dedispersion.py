import numpy as np
import ctypes
import multiprocessing


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


    # FIXME: at small ``dt`` it uses too small DM-step for my laptop RAM:)
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
    # MHz ** 2 * cm ** 3 * s / pc
    k = 1. / (2.410331 * 10 ** (-4))
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


def multi_wrapper(func):
    def wrap(args):
        return func(*args)
    return wrap


def noncoherent_dedisperse(dsp, dm_grid, nu, nu_max, d_t, savefig=None, threads=1):
    """
    Method that de-disperse ``Frame`` instance with range values of
    dispersion measures and average them in frequency to obtain image in
    (t, DM)-plane.
    :param dm_grid:
        Array-like of value of DM on which to de-disperse [cm^3/pc].
    :param savefig: (optional)
        File to save picture.
    :param threads: (optional)
        Number of threads used for parallelization with ``multiprocessing``
        module. If > 1 then it isn't used. (default: 1)
    """
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
    :return:
        Shared array.
    """
    # Using shared array (http://stackoverflow.com/questions/5549190 by pv.)
    i, j = array.shape
    shared_array_base = multiprocessing.Array(ctypes.c_float, i * j)
    shared_array =  np.ctypeslib.as_array(shared_array_base.get_obj()).reshape((i, j))
    shared_array += dsp.copy()
    return shared_array

if __name__ == '__main__':
    from fits_io import get_dyn_spectr
    idi_fits = '/mnt/frb_data/raw_data/re03jy/RE03JY_EF_C_AUTO.idifits'
    t, nu_array, dsp = get_dyn_spectr(idi_fits, time=slice(0, 600000),
                                      complex_indx=0, stokes_indx=0)
    dsp += get_dyn_spectr(idi_fits, time=slice(0, 600000), complex_indx=0,
                          stokes_indx=1)[2]
    dsp *= 0.5
    nu = np.reshape(nu_array, 128) / 10 ** 6
    nu_max = np.max(nu_array.ravel()) / 10 ** 6
    d_nu = (nu_array[0][1:] - nu_array[0][:-1])[0] / 10 ** 6
    d_t = (t[1] - t[0]).sec
    ddsp_kwargs = dict()
    ddsp_kwargs.update({'nu_max': nu_max, 'd_nu': d_nu, 'd_t': d_t})
    from frames import Frame
    frame = Frame(128, len(t), nu_max, 0., 32./128, d_t)
    frame.add_values(dsp)
    for i in range(10):
        frame.add_pulse(50. * (i + 1), 5. / 2. ** i, 0.001, dm=500.)
    dm_grid = np.arange(0., 1000., 50.)
    print("Dedispersing...")
    tdm = noncoherent_dedisperse(frame.values, dm_grid, nu, nu_max, d_t,
                                 threads=4)
    print("Searching candidates...")
    search_kwargs = {'threshold': 99.85, 'n_d_x': 3., 'n_d_y': 15.}
    from search import search_candidates
    candidates = search_candidates(tdm, **search_kwargs)
    for candidate in candidates:
        print (t[candidate['max_pos'][1]] - t[0]).sec,\
            dm_grid[candidate['max_pos'][0]]
        print "Max: ", tdm[candidate['max_pos'][0], candidate['max_pos'][1]]
        print "dx: ", candidate['dx']
        print "dy: ", candidate['dy']
