# -*- coding: utf-8 -*-
# from candidates import Candidate


# TODO: Put dynamical spectra with it's metada to class (a-la ``frames.Frame``)?
# Then we initialize ``Searcher`` instance with instance of this class. If so
# then Sanya should implement creation instances of this new dynamical spectra
# class with his code
# FIXME: What if i want to process dynamical spectra with different algorithms?
# Then one can initialize many ``Searcher`` instances with different
# ``search_func``, it's arguments, optionally different ``preprocess_func`` and
# it's arguments and put metadata about algorithms with it's parameters to
# ``meta_data``. This is `Strategy`` pattern in OOP. But if some search
# algorithms use the same ``preprocess_func`` and it's arguments then we have to
# redo preprocessing each time. Thus i need implement different search with the
# same preprocessed data.
class Searcher(object):
    """
    Basic class that handles searching candidates in dynamical spectra.

    :param dsp:
        2D numpy array with dynamical spectra.
    :param de_disp_func:
        Function that de-disperse dynamical spectra.
    :param search_func:
        Function that used to search candidates in optionally preprocessed
        dynamical spectra and returns
    :param meta_data:
        Dictionary with metadata describing current dynamical spectra.
    :param preprocess_func: (optional)
        Function that optionally preprocesses dynamical spectra (e.g.
        de-dispersion).
    :param de_disp_args: (optional)
        A list of optional positional arguments for ``de_disp_func``.
        ``de_disp_func`` will be called with the sequence ``de_disp_func(dsp,
        *de_disp_args, **de_disp_kwargs)``.
    :param de_disp_kwargs: (optional)
        A list of optional keyword arguments for ``de_disp_func``.
        ``de_disp_func`` will be called with the sequence ``de_disp_func(dsp,
        *de_disp_args, **de_disp_kwargs)``.
    :param search_args: (optional)
        A list of optional positional arguments for ``search_func``.
        ``search_func`` will be called with the sequence ``search_func(dsp,
        *search_args, **search_kwargs)``.
    :param search_kwargs: (optional)
        A list of optional keyword arguments for ``search_func``.
        ``search_func`` will be called with the sequence ``search_func(dsp,
        *search_args, **search_kwargs)``.
    :param preprocess_args: (optional)
        A list of optional positional arguments for ``preprocess_func``.
        ``preprocess_func`` will be called with the sequence
        ``preprocess_func(dsp, *preprocess_args, **preprocess_kwargs)``.
    :param preprocess_kwargs: (optional)
        A list of optional keyword arguments for ``preprocess_func``.
        ``preprocess_func`` will be called with the sequence
        ``preprocess_func(dsp, *preprocess_args, **preprocess_kwargs)``.
    """
    def __init__(self, dsp, de_disp_func, search_func, meta_data,
                 preprocess_func=None, de_disp_args=[], de_disp_kwargs={},
                 search_args=[], search_kwargs={}, preprocess_args=[],
                 preprocess_kwargs={}):
        self.dsp = dsp
        self.meta_data = meta_data
        self._search_func = search_func
        self._de_disp_func = de_disp_func
        self.de_disp_args = de_disp_args
        self.de_disp_kwargs = de_disp_kwargs
        self.search_args = search_args
        self.search_kwargs = search_kwargs
        self._preprocess_func = preprocess_func
        self.preprocess_args = preprocess_args
        self.preprocess_kwargs = preprocess_kwargs
        self.de_disp_func = _function_wrapper(self._de_disp_func,
                                              self.de_disp_args,
                                              self.de_disp_kwargs)
        self.search_func = _function_wrapper(self._search_func,
                                             self.search_args,
                                             self.search_kwargs)
        if preprocess_func is not None:
            self.preprocess_func = _function_wrapper(self._preprocess_func,
                                                     self.preprocess_args,
                                                     self.preprocess_kwargs)
        else:
            self.preprocess_func = None

    def _preprocess(self):
        result = self.preprocess_func()
        return result

    def _de_disperse(self):
        result = self.de_disp_func(self.dsp)
        return result

    def search(self):
        """
        Search candidates in optionally preprocessed dynamical spectra.

        :return:
            List of ``Candidates`` instances.
        """
        de_dispersed_dsp = self._de_disperse()
        if self.preprocess_func is not None:
            to_search = self.preprocess_func(de_dispersed_dsp)
        else:
            to_search = de_dispersed_dsp
        candidates = self.search_func(to_search)
        return candidates


def train_classifyer(dsp, amps, widths, dms, times=None):
    """
    Train classifier using artificially injected pulses in small time
    interval of dynamical spectra.

    :param dsp:
        Dynamical spectra to use.
    :param amps:
        Iterable of true pulse amplitudes.
    :param widths:
        Iterable of true pulse widths.
    :param times: (optional)
        Iterable of pulse's arrival times. If ``None`` then randomly put pulses
        in dynamical spectra.

    :return:
        Trained instance of classifier.

    :notes:
        Several steps are maid:
        * Artificial pulses are injected into dynamical spectra
        * Using some pre-processing of dynamical spectra and search algorithm
        (through ``Searcher`` class) injected pulses are found
    """
    for pars in zip(times, amps, widths, dms):
        print "Adding pulse wih t0, amp, width, DM = ", pars
        frame.add_pulse(*pars)


class _function_wrapper(object):
    """
    This is a hack to make the functions pickleable when ``args`` or ``kwargs``
    are also included.

    """
    def __init__(self, f, args, kwargs):
        self.f = f
        self.args = args
        self.kwargs = kwargs

    def __call__(self, x):
        try:
            return self.f(x, *self.args, **self.kwargs)
        except:
            import traceback
            import inspect
            print("frb: Exception while calling your function:",
                  inspect.stack()[0][3])
            print("  params:", x)
            print("  args:", self.args)
            print("  kwargs:", self.kwargs)
            print("  exception:")
            traceback.print_exc()
            raise


if __name__ == '__main__':
    # Use case
    import numpy as np

    print "Creating Dynamical Spectra"
    from frames import Frame
    frame = Frame(256, 10000, 1684., 0., 16./256, 1./1000)
    print "Adding pulse"
    frame.add_pulse(1., 0.15, 0.003, 100.)
    frame.add_pulse(2., 0.15, 0.003, 200.)
    frame.add_pulse(3., 0.15, 0.003, 300.)
    frame.add_pulse(4., 0.15, 0.003, 500.)
    frame.add_pulse(5., 0.15, 0.003, 700.)
    print "Adding noise"
    frame.add_noise(0.5)

    meta_data = {'antenna': 'WB', 'freq': 1684., 'band': 'L', 'pol': 'R',
                 'exp_code': 'raks00'}
    from dedispersion import de_disperse
    from search import search_candidates, create_ellipses
    dm_grid = np.arange(0., 1000., 25.)
    searcher = Searcher(dsp=frame.values, de_disp_func=de_disperse,
                        search_func=search_candidates, meta_data=meta_data,
                        preprocess_func=create_ellipses,
                        de_disp_args=[dm_grid],
                        de_disp_kwargs={'nu_max': 1684., 'd_nu': 16./256,
                                        'd_t': 1./1000},
                        search_kwargs={'n_d_x': 3., 'n_d_y': 10.},
                        preprocess_kwargs={'disk_size': 5,
                                           'threshold_perc': 99.8})
    candidates = searcher.search()
