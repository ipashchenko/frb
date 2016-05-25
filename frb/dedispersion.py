# -*- coding: utf-8 -*-

# MHz ** 2 * cm ** 3 * s / pc
k = 1. / (2.410331 * 10 ** (-4))


def de_disperse_cumsum(dsp, dm_values):
    """
    De-disperse dynamical spectra with grid of user specifies values of DM.

    :param dsp:
        Instance of ``DynSpectra`` class.
    :param dm_values:
        Array-like of DM values to de-disperse [cm^3 /pc].

    :return:
        2D numpy array (a.k.a. TDM-array) (#DM, #t)

    :notes:
        Probably, it won't work (at least efficiently) when time shift between
        close frequency channels > one time interval.
    """
    return dsp.de_disperse_cumsum(dm_values)


def noncoherent_dedisperse(dsp, dm_grid, threads=1):
    """
    Method that de-disperse dynamical spectra with range values of dispersion
    measures and average them in frequency to obtain image in (t, DM)-plane.

    :param dsp:
        Instance of ``DynSpectra`` class.
    :param dm_grid:
        Array-like of value of DM on which to de-disperse [cm^3/pc].
    :param threads: (optional)
        Number of threads used for parallelization with ``multiprocessing``
        module. If ``1`` then it isn't used. (default: 1)
    """
    return dsp.grid_dedisperse(dm_grid, threads=threads)

