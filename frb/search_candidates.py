# -*- coding: utf-8 -*-
from candidates import SearchedData
import hashlib
import numpy as np
from astropy.time import TimeDelta
from queries import connect_to_db


class Searcher(object):
    """
    Basic class that handles searching candidates in dynamical spectra.

    :param dsp:
        2D numpy array with dynamical spectra.
    :param meta_data:
        Dictionary with metadata describing current dynamical spectra. It must
        include ``exp_name`` [string], ``antenna`` [string], ``freq`` [string],
        ``band`` [string], ``pol`` [string], ``t_0`` [astropy.time.Time],
        ``nu_max`` [number, MHz], ``d_nu`` [number, MHz], ``d_t`` [number, s]
        keys.

        Eg. {'exp_name': 'raks03ra', 'antenna': 'AR'. 'freq': 'L', 'band': 'U',
        'pol': 'L', 't_0': ``instance of astropy.time.Time``, 'nu_max':
        ``1684.0``, 'd_nu': ``0.5``, 'd_t': ``0.001``}
    """
    def __init__(self, dsp, meta_data):
        self.dsp = dsp

        # Parsing meta-data
        n_nu, n_t = np.shape(dsp)
        self.n_nu = n_nu
        self.n_t = n_t
        self.t_0 = meta_data.get('t_0')
        self.d_nu = meta_data.get('d_nu')
        d_t = meta_data.get('d_t')
        self.t_end = self.t_0 + n_t * TimeDelta(d_t, format='sec')
        self.d_t = d_t
        self.nu_max = meta_data.get('nu_max')

        self.meta_data = meta_data
        self.meta_data.update({'t_end': self.t_end.utc.datetime,
                               't_0': self.t_0.utc.datetime})

        self._de_dispersed_cache = dict()
        self._preprocessed_cache = dict()

        self._de_dispersed_data = None
        # This contain md5-sum for current de-dispersion parameters. We need
        # this if want to cache de-dispersion + pre-processing steps
        self._de_disp_m = None

        self._pre_processed_data = None

    def de_disperse(self, de_disp_func, *args, **kwargs):
        kwargs.update({'nu_max': self.nu_max, 'd_nu': self.d_nu,
                       'd_t': self.d_t})
        # Sort kwargs keys before checking in cache
        kwargs = {key: kwargs[key] for key in sorted(kwargs)}
        m = hashlib.md5()
        margs = [x.__repr__() for x in args]
        mkwargs = [x.__repr__() for x in kwargs.values()]
        map(m.update, margs + mkwargs)
        m.update(de_disp_func.__name__)
        key = m.hexdigest()
        result = self._de_dispersed_cache.get(key, None)
        if result is not None:
            print "Found cached de-dispersed data..."
        else:
            result = de_disp_func(self.dsp, *args, **kwargs)
            # Put to cache
            self._de_dispersed_cache[key] = result
        self._de_dispersed_data = result
        self._de_disp_m = m.copy()

    def reset_pre_processing(self):
        self._pre_processed_data = None

    def reset_dedispersion(self):
        self._de_dispersed_data = None

    def pre_process(self, preprocess_func, *args, **kwargs):
        # Will only search for cached values with the same de-dispersion & pre-
        # processing parameters
        # Sort kwargs keys before checking in cache
        kwargs = {key: kwargs[key] for key in sorted(kwargs)}
        m = self._de_disp_m.copy()
        margs = [x.__repr__() for x in args]
        mkwargs = [x.__repr__() for x in kwargs.values()]
        map(m.update, margs + mkwargs)
        # If no pre-processing is supposed => just pass data
        if preprocess_func is None:
                result = self._de_dispersed_data
        else:
            m.update(preprocess_func.__name__)
            key = m.hexdigest()
            result = self._preprocessed_cache.get(key, None)
            if result is not None:
                print "Found cached preprocessed data..."
            else:
                result = preprocess_func(self._de_dispersed_data, *args,
                                         **kwargs)
                self._preprocessed_cache[key] = result

        self._pre_processed_data = result

    def search(self, search_func, *args, **kwargs):
        """
        Search candidates in optionally preprocessed dynamical spectra.

        :return:
            List of ``Candidate`` instances.
        """
        kwargs.update({'t_0': self.t_0, 'd_t': self.d_t})
        candidates = search_func(self._pre_processed_data, *args, **kwargs)

        return candidates

    def run(self, de_disp_func, search_func=None, preprocess_func=None,
            de_disp_args=[], de_disp_kwargs={}, search_args=[],
            search_kwargs={}, preprocess_args=[], preprocess_kwargs={},
            db_file=None):
        """

        :param de_disp_func:
            Function that used to de-disperse dynamical spectra.
        :param search_func:
            Function that used to search candidates in optionally preprocessed
            dynamical spectra and returns list of ``Candidate`` instances.
        :param preprocess_func: (optional)
            Function that optionally preprocesses de-dispersed dynamical
            spectra. If ``None`` then don't use preprocessing. (default:
            ``None``)
        :param de_disp_args: (optional)
            A list of optional positional arguments for ``de_disp_func``.
            ``de_disp_func`` will be called with the sequence
            ``de_disp_func(dsp, *de_disp_args, **de_disp_kwargs)``.
        :param de_disp_kwargs: (optional)
            A list of optional keyword arguments for ``de_disp_func``.
            ``de_disp_func`` will be called with the sequence
            ``de_disp_func(dsp, *de_disp_args, **de_disp_kwargs)``.
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

        :return:
            List of ``Candidate`` instances.

        :note:
            When running through ``Searcher.run`` method it saves metadata on
            processed dynamical spectra and candidates found to DB.
        """

        self.de_disperse(de_disp_func, *de_disp_args, **de_disp_kwargs)
        self.pre_process(preprocess_func, *preprocess_args, **preprocess_kwargs)
        candidates = self.search(search_func, *search_args, **search_kwargs)

        # Save to DB metadata of dsp
        algo = 'de_disp_{}_{}_{} pre_process_{}_{}_{}' \
               ' search_{}_{}_{}'.format(de_disp_func.__name__, de_disp_args,
                                         de_disp_kwargs,
                                         preprocess_func.__name__,
                                         preprocess_args, preprocess_kwargs,
                                         search_func.__name__, search_args,
                                         search_kwargs)
        searched_data = SearchedData(algo=algo, **self.meta_data)
        searched_data.candidates = candidates
        # Saving searched meta-data and found candidates to DB
        if db_file is not None:
            session = connect_to_db(db_file)
            session.add(searched_data)
            session.commit()

        return candidates

