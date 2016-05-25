import multiprocessing
import ctypes
import numpy as np
import pickle_method
from utils import vint, vround, save_hdf5, read_hdf5
from astropy.time import Time, TimeDelta

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


class MetaData(dict):
    """
    Class that describes RA experiment metadata.

    See http://stackoverflow.com/questions/2060972/subclassing-python-dictionary-to-override-setitem
    """

    meta_keys = ['exp_code', 'antenna', 'freq', 'band', 'pol']
    required_keys = meta_keys
    meta_values = {'freq': ('k', 'c', 'l', 'p'), 'band': ('u', 'l', 'ul'),
                   'pol': ('l', 'r', 'lr')}

    def __init__(self, *args, **kwargs):
        super(MetaData, self).__init__()
        self.update(*args, **kwargs)
        for key in self.required_keys:
            if key not in self:
                raise Exception("Absent key {} in metadata".format(key))

    def __setitem__(self, key, value):
        # optional processing here
        if key not in self.meta_keys:
            raise Exception("Not allowed key in metadata: {}".format(key))
        if key in self.meta_values and value not in self.meta_values[key]:
            raise Exception("Not allowed value: {} in"
                            " metadata for key: {}".format(value, key))
        super(MetaData, self).__setitem__(key, value)

    def update(self, *args, **kwargs):
        if args:
            if len(args) > 1:
                raise TypeError("update expected at most 1 arguments, "
                                "got {}".format(len(args)))
            other = dict(args[0])
            for key in other:
                self[key] = other[key]
        for key in kwargs:
            self[key] = kwargs[key]

    def setdefault(self, key, value=None):
        if key not in self:
            self[key] = value
        return self[key]


class DynSpectra(object):
    """
    Basic class that represents a set of regularly spaced frequency channels
    with regularly measured values (time sequence of autospectra).

    :param n_nu:
        Number of spectral channels.
    :param n_t:
        Number of time steps.
    :param nu_0:
        Frequency of highest frequency channel [MHz].
    :param dnu:
        Width of spectral channel [MHz].
    :param d_t:
        Time step [s].
    :param meta_data:
        Dictionary with metadata describing current dynamical spectra. It must
        include ``exp_code`` [string], ``antenna`` [string], ``freq`` [string],
        ``band`` [string], ``pol`` [string] keys.
        Eg. {'exp_name': 'raks03ra', 'antenna': 'AR'. 'freq': 'L', 'band': 'U',
        'pol': 'L'}
    :param t_0: (optional)
        Time of first measurement. Instance of ``astropy.time.Time`` class. If
        ``None`` then use time of initialization. (default: ``None``)

    """


    def __init__(self, n_nu, n_t, nu_0, d_nu, d_t, meta_data=None, t_0=None):
        self.n_nu = n_nu
        self.n_t = n_t
        self.nu_0 = nu_0
        self.t_0 = t_0 or Time.now()
        # Using shared array (http://stackoverflow.com/questions/5549190 by pv.)
        shared_array_base = multiprocessing.Array(ctypes.c_float, n_nu * n_t)
        self.values =\
            np.ctypeslib.as_array(shared_array_base.get_obj()).reshape((n_nu,
                                                                        n_t,))

        nu = np.arange(n_nu)
        t = np.arange(n_t)
        self.nu = (nu_0 - nu * d_nu)[::-1]
        self.d_t = TimeDelta(d_t, format='sec')
        self.t = self.t_0 + t * self.d_t
        self.t_end = self.t[-1]
        self.d_nu = d_nu
        self.meta_data = MetaData(meta_data)

    @property
    def shape(self):
        """
        Length of time [s] and frequency [MHz] dimensions.
        """
        return self.n_t * self.d_t.sec, self.n_nu * self.d_nu

    def add_values(self, array):
        """
        Add dyn. spectra in form of numpy array (#ch, #t,) to instance.

        :param array:
            Array-like of dynamical spectra (#ch, #t,).
        """
        array = np.atleast_2d(array)
        assert self.values.shape == array.shape
        self.values += array

    # FIXME: Handle start time in slices somehow
    def slice(self, t_start, t_stop):
        """
        Slice frame using specified fractions of time interval.

        :param t_start:
            Number [0, 1] - fraction of total time interval.
        :param t_stop:
            Number [0, 1] - fraction of total time interval.

        :return:
            Instance of ``DynSpectra`` class.
        """
        assert t_start < t_stop
        frame = DynSpectra(self.n_nu, int(round(self.n_t * (t_stop - t_start))),
                           self.nu_0, self.d_nu, self.d_t,
                           meta_data=self.meta_data, t_0=self.t_0)
        frame.add_values(self.values[:, int(t_start * self.n_t): int(t_stop *
                                                                     self.n_t)])
        return frame

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
        nt_all = vint(vround(dt_all / self.d_t.sec))
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
        nt_all = vint(vround(dt_all / self.d_t.sec))
        # Container for summing de-dispersed frequency channels
        values = np.zeros(self.n_t)
        # Roll each axis (freq. channel) to each own number of time steps.
        for i in range(self.n_nu):
            values += np.roll(self.values[i], -nt_all[i])

        return values / self.n_nu

    def de_disperse_cumsum(self, dm_values):
        """
        De-disperse dynamical spectra with grid of user specifies values of DM.

        :param dm_values:
            Array-like of DM values to de-disperse [cm^3 /pc].

        :return:
            2D numpy array (a.k.a. TDM-array) (#DM, #t)

        :notes:
            Probably, it won't work (at least efficiently) when time shift
            between close frequency channels > one time interval.
        """
        # MHz ** 2 * cm ** 3 * s / pc
        k = 1. / (2.410331 * 10 ** (-4))

        # Frequency of highest frequency channel [MHz].
        nu_max = self.nu_0
        # Time step [s].
        d_t = self.d_t.sec
        dm_values = np.array(dm_values)
        n_nu = self.n_nu
        n_t = self.n_t
        nu = self.nu
        # Pre-calculating cumulative sums and their difference
        cumsums = np.ma.cumsum(self.values[::-1, :], axis=0)
        dcumsums = np.roll(cumsums, 1, axis=1) - cumsums

        # Calculate shift of time caused by de-dispersion for all channels and
        # all values of DM
        dt_all = k * dm_values[:, np.newaxis] * (1. / nu ** 2. -
                                                 1. / nu_max ** 2.)
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

    def add_pulse(self, t_0, amp, width, dm):
        """
        Add pulse to frame.

        :param t_0:
            Arrival time of pulse at highest frequency channel [s]. Counted
            from start time of ``DynSpectra`` instance.
        :param amp:
            Amplitude of pulse.
        :param width:
            Width of gaussian pulse [s] (in time domain).
        :param dm: (optional)
            Dispersion measure of pulse [cm^3 / pc].

        """
        t_0 = TimeDelta(t_0, format='sec')

        # MHz ** 2 * cm ** 3 * s / pc
        k = 1. / (2.410331 * 10 ** (-4))

        # Calculate arrival times for all channels
        t0_all = (t_0.sec * np.ones(self.n_nu)[:, np.newaxis] +
                  k * dm * (1. / self.nu ** 2. -
                            1. / self.nu_0 ** 2.))[0]
        pulse = amp * np.exp(-0.5 * ((self.t - self.t_0).sec -
                                     t0_all[:, np.newaxis]) ** 2 / width ** 2.)
        self.values += pulse

    def rm_pulse(self, t_0, amp, width, dm):
        """
        Remove pulse to frame.

        :param t_0:
            Arrival time of pulse at highest frequency channel [s]. Counted
            from start time of ``DynSpectra`` instance.
        :param amp:
            Amplitude of pulse.
        :param width:
            Width of gaussian pulse [s] (in time domain).
        :param dm:
            Dispersion measure of pulse [cm^3 / pc].

        """
        self.add_pulse(t_0, -amp, width, dm)

    def add_noise(self, std):
        """
        Add noise to frame using specified rayleigh-distributed noise.

        :param std:
            Std of rayleigh-distributed uncorrelated noise.
        """
        noise =\
            np.random.rayleigh(std,
                               size=(self.n_t *
                                     self.n_nu)).reshape(np.shape(self.values))
        self.values += noise

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
        Method that de-disperse ``DynSpectra`` instance with range values of
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

    def save_to_hdf5(self, fname, name='dsp'):
        """
        Save data to HDF5 format.

        :param fname:
            File to save data.
        :param name: (optional)
            Name of dataset to use. (default: ``dsp``)

        :note:
            HDF5 hasn't time formats. Using ``str(datetime)`` to create strings
            with microseconds.
        """
        import h5py
        f = h5py.File(fname, "w")
        dset = f.create_dataset(name, data=self.values, chunks=True,
                                compression='gzip')
        meta_data = self.meta_data.copy()
        meta_data.update({'n_nu': self.n_nu, 'n_t': self.n_t, 'nu_0': self.nu_0,
                          'd_nu': self.d_nu, 'd_t': self.d_t.sec,
                          't_0': str(self.t_0)})
        for key, value in meta_data.items():
            dset.attrs[key] = value
        f.flush()
        f.close()

def create_from_hdf5(fname, name='dsp', n_nu_discard=0):
    """
    Function that creates instance of ``DynSpectra`` class from HDF5-file.

    :param fname:
        Name of HDF5-file with `dsp` data set that is 2D numpy.ndarray with rows
        representing frequency channels and columns - 1d-time series of data for
        each frequency channel and meta-data.
    :param name: (optional)
        Name of dataset to use. (default: ``dsp``)
    :param n_nu_discard: (optional)
        NUmber of spectral channels to discard symmetrically from both low and
         high frequency.

    :return:
        Instance of ``DynSpectra`` class.
    """
    data, meta_data = read_hdf5(fname, name)
    n_nu = meta_data.pop('n_nu')
    n_t = meta_data.pop('n_t')
    nu_0 = meta_data.pop('nu_0')
    d_nu = meta_data.pop('d_nu')
    d_t = meta_data.pop('d_t')
    t_0 = Time(meta_data.pop('t_0'))
    dsp = DynSpectra(n_nu - n_nu_discard, n_t, nu_0 - n_nu_discard * d_nu / 2.,
                     d_nu, d_t, meta_data, t_0=t_0)
    dsp.add_values(data)
    return dsp


def create_from_txt(fname, nu_0, d_nu, d_t, meta_data, t_0=None,
                    n_nu_discard=0):
    """
    Function that creates instance of ``DynSpectra`` class from text file.

    :param fname:
        Name of txt-file with rows representing frequency channels and columns -
        1d-time series of data for each frequency channel.
    :param nu_0:
        Frequency of highest frequency channel [MHz].
    :param d_nu:
        Width of spectral channel [MHz].
    :param d_t:
        Time step [s].
    :param meta_data:
        Dictionary with metadata describing current dynamical spectra. It must
        include ``exp_name`` [string], ``antenna`` [string], ``freq`` [string],
        ``band`` [string], ``pol`` [string] keys.
    :param t_0: (optional)
        Time of first measurement. Instance of ``astropy.time.Time`` class. If
        ``None`` then use time of initialization. (default: ``None``)
    :param n_nu_discard: (optional)
        NUmber of spectral channels to discard symmetrically from both low and
         high frequency.

    :return:
        Instance of ``DynSpectra`` class.
    """
    assert not int(n_nu_discard) % 2

    try:
        values = np.load(fname).T
    except IOError:
        values = np.loadtxt(fname, unpack=True)
    n_nu, n_t = np.shape(values)
    dsp = DynSpectra(n_nu - n_nu_discard, n_t, nu_0 - n_nu_discard * d_nu / 2.,
                     d_nu, d_t, meta_data=meta_data, t_0=t_0)
    if n_nu_discard:
        dsp.values += values[n_nu_discard / 2: -n_nu_discard / 2, :]
    else:
        dsp.values += values

    return dsp


if __name__ == '__main__':
    # Creating fake dynamical spectra
    dsp = DynSpectra(128, 12000, 1684., 16. / 256, 1. / 1000,
                     meta_data={'antenna': 'RA', 'exp_code': 'raks100',
                                  'freq': 'l', 'band': 'u', 'pol': 'r'})
    dsp.add_pulse(1., 0.2, 0.003, 100.)
    dsp.add_pulse(2., 0.2, 0.003, 200.)
    dsp.add_pulse(3., 0.2, 0.003, 300.)
    dsp.add_pulse(4., 0.2, 0.003, 500.)
    dsp.add_pulse(5., 0.2, 0.003, 700.)
    dsp.add_noise(0.2)
    # Reading from txt
    txt = '/home/ilya/code/akutkin/frb/data/100_sec_wb_raes08a_128ch.asc'
    dsp_t = create_from_txt(txt, 1684., 16. / 128, 0.001,
                            meta_data={'antenna': 'RA', 'exp_code': 'raks100',
                                         'freq': 'l', 'band': 'u', 'pol': 'r'})
    # Slicing
    dsp1 = dsp_t.slice(0, 0.1)
    dsp2 = dsp_t.slice(0.1, 0.9)
    dsp3 = dsp_t.slice(0.9, 1)
    # Saving/reading HDF5
    dsp_t.save_to_hdf5('test.hdf5')
    new_dsp = create_from_hdf5('test.hdf5')
