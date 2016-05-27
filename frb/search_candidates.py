# -*- coding: utf-8 -*-
import os
import h5py
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

    :param cache_dir: (optional)
        Directory to store cache HDF5 files. If ``None`` - use CWD. (default:
        ``None``)
    """
    def __init__(self, dsp, cache_dir=None):
        self.dsp = dsp
        self.meta_data = dsp.meta_data.copy()

        # This needed for string conversion
        self.meta_data.update({'t_end': self.dsp.t_end.utc.datetime,
                               't_0': self.dsp.t_0.utc.datetime})

        if cache_dir is None:
            cache_dir = os.getcwd()
        self.cache_dir = cache_dir

        self._de_disp_cache_fname = os.path.join(cache_dir,
                                                 self._cache_fname_prefix +
                                                 "_dedisp.hdf5")
        self._pre_proces_cache_fname = os.path.join(cache_dir,
                                                    self._cache_fname_prefix +
                                                    "_preproc.hdf5")
        self._de_dispersed_cache = h5py.File(self._de_disp_cache_fname)
        self._preprocessed_cache = h5py.File(self._pre_proces_cache_fname)

        self._de_dispersed_data = None
        # This contain md5-sum for current de-dispersion parameters. We need
        # this if want to cache de-dispersion + pre-processing steps
        self._de_disp_m = None

        self._pre_processed_data = None

    # TODO: Add n_nu, n_t, d_nu, d_t, nu_max
    @property
    def _cache_fname_prefix(self):
        date_0, time_0 = str(self.meta_data['t_0']).split(' ')
        date_1, time_1 = str(self.meta_data['t_end']).split(' ')
        return "{}_{}_{}_{}_{}_{}_{}".format(self.meta_data['exp_code'],
                                             self.meta_data['antenna'],
                                             self.meta_data['freq'], date_0,
                                             time_0, date_1, time_1)

    def de_disperse(self, de_disp_func, *args, **kwargs):
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
            result = result.value
        else:
            result = de_disp_func(self.dsp, *args, **kwargs)
            # Put to cache
            self._de_dispersed_cache.create_dataset(key, data=result,
                                                    chunks=True,
                                                    compression='gzip')
            self._de_dispersed_cache.flush()
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
                result = result.value
            else:
                result = preprocess_func(self._de_dispersed_data.copy(), *args,
                                         **kwargs)
                self._preprocessed_cache.create_dataset(key, data=result,
                                                        chunks=True,
                                                        compression='gzip')
                self._preprocessed_cache.flush()

        self._pre_processed_data = result

    def search(self, search_func, *args, **kwargs):
        """
        Search candidates in optionally preprocessed dynamical spectra.

        :return:
            List of ``Candidate`` instances.
        """
        kwargs.update({'t_0': self.dsp.t_0,
                       'd_t': self.dsp.d_t})
        candidates = search_func(self._pre_processed_data.copy(), *args,
                                 **kwargs)

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
               ' search_{}_{}'.format(de_disp_func.__name__, de_disp_args,
                                         de_disp_kwargs,
                                         preprocess_func.__name__,
                                         preprocess_args, preprocess_kwargs,
                                         search_func.__name__, search_kwargs)
        searched_data = SearchedData(algo=algo, **self.meta_data)
        searched_data.candidates = candidates
        # Saving searched meta-data and found candidates to DB
        if db_file is not None:
            session = connect_to_db(db_file)
            session.add(searched_data)
            session.commit()

        return candidates

