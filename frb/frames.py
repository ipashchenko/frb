import multiprocessing
import ctypes
import numpy as np
import pickle_method
from utils import vint, vround, delta_dm_max

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


class Frame(object):
    """
    Basic class that represents a set of regularly spaced frequency channels
    with regularly measured values (time sequence of autospectra).

    :param n_nu:
        Number of spectral channels.
    :param n_t:
        Number of time steps.
    :param nu_0:
        Frequency of highest frequency channel [MHz].
    :param t_0:
        Time of first measurement.
    :param dnu:
        Width of spectral channel [MHz].
    :param dt:
        Time step [s].

    """
    def __init__(self, n_nu, n_t, nu_0, t_0, dnu, dt):
        self.n_nu = n_nu
        self.n_t = n_t
        self.nu_0 = nu_0
        self.t_0 = t_0
        # Using shared array (http://stackoverflow.com/questions/5549190 by pv.)
        shared_array_base = multiprocessing.Array(ctypes.c_float, n_nu * n_t)
        self.values = np.ctypeslib.as_array(shared_array_base.get_obj()).reshape((n_nu,
                                                                                  n_t,))

        nu = np.arange(n_nu)
        t = np.arange(n_t)
        self.nu = (nu_0 - nu * dnu)[::-1]
        self.t = t_0 + t * dt
        self.dt = dt
        self.dnu = dnu

    def add_values(self, array):
        """
        Add dyn. spectra in form of numpy array (#ch, #t,) to instance.

        :param array:
            Array-like of dynamical spectra (#ch, #t,).
        """
        array = np.atleast_2d(array)
        assert self.values.shape == array.shape
        self.values += array

    def slice(self, channels, times):
        """
        Slice frame using specified channels and/or times.
        """
        raise NotImplementedError

    def _de_disperse_by_value(self, dm):
        """
        De-disperse frame using specified value of DM.

        :param dm:
            Dispersion measure to use in de-dispersion [cm^3 / pc].
        """
        # MHz ** 2 * cm ** 3 * s / pc
        k = 1. / (2.410331 * 10 ** (-4))

        # Calculate shift of time caused by de-dispersion for all channels
        dt_all = k * dm * (1. / self.nu ** 2. - 1. / self.nu_0 ** 2.)
        # Find what number of time bins corresponds to this shifts
        nt_all = vint(vround(dt_all / self.dt))
        # Roll each axis (freq. channel) to each own number of time steps.
        values = list()
        for i in range(self.n_nu):
            values.append(np.roll(self.values[i], -nt_all[i]))
        values = np.vstack(values)

        return values

    def _de_disperse_by_value_freq_average(self, dm):
        """
        De-disperse frame using specified value of DM and average in frequency.

        :param dm:
            Dispersion measure to use in de-dispersion [cm^3 / pc].

        :note:
            This method avoids creating ``(n_nu, n_t)`` arrays and must be
            faster for data with big sizes. But it returns already frequency
            averaged de-dispersed dyn. spectra.

        """
        # MHz ** 2 * cm ** 3 * s / pc
        k = 1. / (2.410331 * 10 ** (-4))

        # Calculate shift of time caused by de-dispersion for all channels
        dt_all = k * dm * (1. / self.nu ** 2. - 1. / self.nu_0 ** 2.)
        # Find what number of time bins corresponds to this shifts
        nt_all = vint(vround(dt_all / self.dt))
        # Container for summing de-dispersed frequency channels
        values = np.zeros(self.n_t)
        # Roll each axis (freq. channel) to each own number of time steps.
        for i in range(self.n_nu):
            values += np.roll(self.values[i], -nt_all[i])

        return values / self.n_nu

    # TODO: if one choose what channels to plot - use ``extent`` kwarg.
    def plot(self, plot_indexes=True, savefig=None):
        if plt is not None:
            plt.figure()
            plt.matshow(self.values, aspect='auto')
            plt.colorbar()
            if not plot_indexes:
                raise NotImplementedError("Ticks haven't implemented yet")
                # plt.xticks(np.linspace(0, 999, 10, dtype=int),
                # frame.t[np.linspace(0, 999, 10, dtype=int)])
            plt.xlabel("time steps")
            plt.ylabel("frequency ch. #")
            plt.title('Dynamical spectra')
            if savefig is not None:
                plt.savefig(savefig, bbox_inches='tight')
            plt.show()

    def add_pulse(self, t_0, amp, width, dm=0.):
        """
        Add pulse to frame.

        :param t_0:
            Arrival time of pulse at highest frequency channel [s].
        :param amp:
            Amplitude of pulse.
        :param width:
            Width of gaussian pulse [s] (in time domain).
        :param dm: (optional)
            Dispersion measure of pulse [cm^3 / pc]. (Default: ``0.``)

        """
        # MHz ** 2 * cm ** 3 * s / pc
        k = 1. / (2.410331 * 10 ** (-4))

        # Calculate arrival times for all channels
        t0_all = (t_0 * np.ones(self.n_nu)[:, np.newaxis] +
                  k * dm * (1. / self.nu ** 2. -
                            1. / self.nu_0 ** 2.))[0]
        pulse = amp * np.exp(-0.5 * (self.t -
                                     t0_all[:, np.newaxis]) ** 2 / width ** 2.)
        self.values += pulse

    def rm_pulse(self, t_0, amp, width, dm=0.):
        """
        Remove pulse to frame.

        :param t_0:
            Arrival time of pulse at highest frequency channel [s].
        :param amp:
            Amplitude of pulse.
        :param width:
            Width of gaussian pulse [s] (in time domain).
        :param dm: (optional)
            Dispersion measure of pulse [cm^3 / pc]. (Default: ``0.``)

        """
        # MHz ** 2 * cm ** 3 * s / pc
        k = 1. / (2.410331 * 10 ** (-4))

        # Calculate arrival times for all channels
        t0_all = (t_0 * np.ones(self.n_nu)[:, np.newaxis] +
                  k * dm * (1. / self.nu ** 2. -
                            1. / self.nu_0 ** 2.))[0]
        pulse = amp * np.exp(-0.5 * (self.t -
                                     t0_all[:, np.newaxis]) ** 2 / width ** 2.)
        self.values -= pulse

    def save_to_txt(self, fname):
        np.savetxt(fname, self.values.T)

    def add_noise(self, std):
        """
        Add noise to frame using specified rayleigh-distributed noise.

        :param std:
            Std of rayleigh-distributed uncorrelated noise.
        """
        noise = np.random.rayleigh(std,
                                   size=(self.n_t *
                                         self.n_nu)).reshape(np.shape(self.values))
        self.values += noise

    # TODO: Check optimal value of ``dm_delta``
    def create_dm_grid(self, dm_min, dm_max, dm_delta=None):
        """
        Method that create DM-grid for current frame.
        :param dm_min:
            Minimal value [cm^3 /pc].
        :param dm_max:
            Maximum value [cm^3 /pc].
        :param dm_delta: (optional)
            Delta of DM for grid [cm^3/pc]. If ``None`` then choose one that
            corresponds to time shift equals to time resolution for frequency
            bandwidth. Actually used value is 5 times larger.
            (default: ``None``)
        :return:
            Numpy array of DM values [cm^3 / pc]
        """
        raise NotImplementedError

    def grid_dedisperse(self, dm_grid, threads=1):
        """
        Method that de-disperse ``Frame`` instance with range values of
        dispersion measures and average them in frequency to obtain image in
        (t, DM)-plane.

        :param dm_grid:
            Array-like of value of DM on which to de-disperse [cm^3/pc].
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

        # Accumulator of de-dispersed frequency averaged frames
        frames = list(m(self._de_disperse_by_value_freq_average,
                        dm_grid.tolist()))
        frames = np.array(frames)

        if pool:
            # Close pool
            pool.close()
            pool.join()

        return frames


class DynSpectra(Frame):
    def __init__(self, n_nu, n_t, nu_0, t_0, dnu, dt, meta_data=None):
        super(DynSpectra, self).__init__(n_nu, n_t, nu_0, t_0, dnu, dt)
        self.meta_data = meta_data
        self.exp_code = meta_data['exp_code']
        self.antenna = meta_data['antenna']
        self.time = meta_data['time']
        self.freq = meta_data['freq']
        self.band = meta_data['band']
        self.pol = meta_data['pol']


def create_from_txt(fname, nu_0, t_0, dnu, dt, n_nu_discard=0):
    """
    Function that creates instance of ``Frame`` class from text file.

    :param fname:
        Name of txt-file with rows representing frequency channels and columns -
        1d-time series of data for each frequency channel.

    :return:
        Instance of ``Frame`` class.
    """
    assert not int(n_nu_discard) % 2

    try:
        values = np.load(fname).T
    except IOError:
        values = np.loadtxt(fname, unpack=True)
    n_nu, n_t = np.shape(values)
    frame = Frame(n_nu - n_nu_discard, n_t,
                  nu_0 - n_nu_discard * dnu / 2., t_0,
                  dnu, dt)
    if n_nu_discard:
        frame.values += values[n_nu_discard / 2 : -n_nu_discard / 2, :]
    else:
        frame.values += values

    return frame


if __name__ == '__main__':
    import time
    print "Creating frame"
    frame = Frame(256, 1200000, 1684., 0., 16./256, 1./1000)
    print "Adding pulse"
    frame.add_pulse(100., 0.15, 0.003, 100.)
    frame.add_pulse(200., 0.15, 0.003, 200.)
    frame.add_pulse(300., 0.15, 0.003, 300.)
    frame.add_pulse(400., 0.15, 0.003, 500.)
    frame.add_pulse(500., 0.15, 0.003, 700.)
    print "Adding noise"
    frame.add_noise(0.5)
