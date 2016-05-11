# -*- coding: utf-8 -*-
from candidates import Candidate, SearchedData
import hashlib
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()


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
    def __init__(self, dsp, meta_data):
        self.dsp = dsp
        self.meta_data = meta_data

        self._de_dispersed_cache = dict()
        self._preprocessed_cache = dict()

        self._de_dispersed_data = None
        # This contain md5-sum for current de-dispersion parameters. We need
        # this if want to cache de-dispersion + pre-processing steps
        self._de_disp_m = None

        self._pre_processed_data = None


    def de_disperse(self, de_disp_func, *args, **kwargs):
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
        candidates = search_func(self._pre_processed_data, *args, **kwargs)

        return candidates

    def run(self, de_disp_func=None, search_func=None, preprocess_func=None,
            de_disp_args=[], de_disp_kwargs={}, search_args=[],
            search_kwargs={}, preprocess_args=[], preprocess_kwargs={}):

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
        searched_data = SearchedData(algo=algo, **meta_data)
        searched_data.candidates = candidates
        # Saving searched meta-data and found candidates to DB
        engine = create_engine("sqlite:////home/ilya/code/akutkin/frb/frb/frb.db",
                               echo=True)
        metadata = Base.metadata
        metadata.create_all(engine)

        from sqlalchemy.orm import sessionmaker
        Session = sessionmaker(bind=engine)
        session = Session()
        session.add(searched_data)
        session.commit()

        return candidates


if __name__ == '__main__':
    # Use case
    import numpy as np

    print "Creating Dynamical Spectra"
    from frames import Frame
    frame = Frame(256, 10000, 1684., 0., 16./256, 1./1000)
    n_pulses = 30
    # Step of de-dispersion
    d_dm = 25.
    print "Adding {} pulses".format(n_pulses)
    np.random.seed(123)
    amps = np.random.uniform(0.1, 0.15, size=n_pulses)
    widths = np.random.uniform(0.001, 0.005, size=n_pulses)
    dm_values = np.random.uniform(0, 1000, size=n_pulses)
    times = np.random.uniform(0., 10., size=n_pulses)
    for t_0, amp, width, dm in zip(times, amps, widths, dm_values):
        frame.add_pulse(t_0, amp, width, dm)
        print "Adding pulse with t0={}, amp={}, width={}, dm={}".format(t_0,
                                                                        amp,
                                                                        width,
                                                                        dm)
    print "Adding noise"
    frame.add_noise(0.5)

    meta_data = {'antenna': 'WB', 'freq': 1684., 'band': 'L', 'pol': 'R',
                 'exp_code': 'raks00'}
    from dedispersion import de_disperse
    from search import search_candidates, create_ellipses
    dm_grid = np.arange(0., 1000., d_dm)
    searcher = Searcher(dsp=frame.values, meta_data=meta_data)
    candidates = searcher.run(de_disp_func=de_disperse,
                              search_func=search_candidates,
                              preprocess_func=create_ellipses,
                              de_disp_args=[dm_grid],
                              de_disp_kwargs={'nu_max': 1684., 'd_nu': 16./256,
                                              'd_t': 1./1000},
                              search_kwargs={'n_d_x': 5., 'n_d_y': 15.,
                                             'd_t': 0.001, 'd_dm': d_dm},
                              preprocess_kwargs={'disk_size': 3,
                                                 'threshold_perc': 98.,
                                                 'statistic': 'mean'})
    print "Found {} pulses".format(len(candidates))
    for candidate in candidates:
        print candidate

    # Using calculated ``Searcher._de_dispersed_data`` & ``_preprocessed_data``
    # FIXME: This is a feature - candidates & searched data won't go to DB when
    # calling ``Searcher.search`` explicitly!
    candidates = searcher.search(search_candidates, n_d_x=8., n_d_y=15.,
                                 d_t=0.001, d_dm=d_dm)
    print "Found {} pulses".format(len(candidates))
    for candidate in candidates:
        print candidate

    # Going through all pipeline & using cached de-dispersed values because
    # preprocessing parameters have changed
    candidates = searcher.run(de_disp_func=de_disperse,
                              search_func=search_candidates,
                              preprocess_func=create_ellipses,
                              de_disp_args=[dm_grid],
                              de_disp_kwargs={'nu_max': 1684., 'd_nu': 16./256,
                                              'd_t': 1./1000},
                              search_kwargs={'n_d_x': 5., 'n_d_y': 15.,
                                             'd_t': 0.001, 'd_dm': d_dm},
                              preprocess_kwargs={'disk_size': 3,
                                                 'threshold_perc': 95.,
                                                 'statistic': 'mean'})
    print "Found {} pulses".format(len(candidates))
    for candidate in candidates:
        print candidate
    dm_grid = np.arange(0., 1000., 50.)
    # Going through all pipeline because even de-dispersion parameters have
    # changed
    candidates = searcher.run(de_disp_func=de_disperse,
                              search_func=search_candidates,
                              preprocess_func=create_ellipses,
                              de_disp_args=[dm_grid],
                              de_disp_kwargs={'nu_max': 1684., 'd_nu': 16./256,
                                              'd_t': 1./1000},
                              search_kwargs={'n_d_x': 5., 'n_d_y': 15.,
                                             'd_t': 0.001, 'd_dm': 50.},
                              preprocess_kwargs={'disk_size': 3,
                                                 'threshold_perc': 95.,
                                                 'statistic': 'mean'})
    print "Found {} pulses".format(len(candidates))
    for candidate in candidates:
        print candidate

    # Going through all pipeline & using cached de-dispersed and pre-processed
    # values
    candidates = searcher.run(de_disp_func=de_disperse,
                              search_func=search_candidates,
                              preprocess_func=create_ellipses,
                              de_disp_args=[dm_grid],
                              de_disp_kwargs={'nu_max': 1684., 'd_nu': 16./256,
                                              'd_t': 1./1000},
                              preprocess_kwargs={'disk_size': 3,
                                                 'threshold_perc': 95.,
                                                 'statistic': 'mean'},
                              search_kwargs={'n_d_x': 9., 'n_d_y': 17.,
                                             'd_t': 0.001, 'd_dm': 50.})
    print "Found {} pulses".format(len(candidates))
    for candidate in candidates:
        print candidate

