# -*- coding: utf-8 -*-
from candidates import Candidate


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
    :param search_func:
        Function that used to search candidates in optionally preprocessed
        dynamical spectra and returns
    :param meta_data:
        Dictionary with metadata describing current dynamical spectra.
    :param preprocess_func: (optional)
        Function that optionally preprocesses dynamical spectra (e.g.
        de-dispersion).
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
    def __init__(self, dsp, search_func, meta_data, preprocess_func=None,
                 search_args=[], search_kwargs={}, preprocess_args=[],
                 prepocess_kwargs={}):
        self.dsp = dsp
        self._search_func = search_func
        self.search_args = search_args
        self.search_kwargs = search_kwargs
        self._preprocess_func = preprocess_func
        self.preprocess_args = preprocess_args
        self.preprocess_kwargs = prepocess_kwargs
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

    def search(self):
        """
        Search candidates in optionally preprocessed dynamical spectra.

        :return:
            List of ``Candidates`` instances.
        """
        if self.preprocess_func is not None:
            to_search = self.preprocess_func(self.dsp)
        else:
            to_search = self.dsp
        candidates = self.search_func(to_search)


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
