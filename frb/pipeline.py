import os
import numpy as np
from astropy.time import TimeDelta
from collections import defaultdict
from dyn_spectra import DynSpectra
from cfx import CFX
from raw_data import M5, dspec_cat
from queries import connect_to_db, query_frb
from search_candidates import Searcher
from dedispersion import de_disperse_cumsum
from search import create_ellipses, search_candidates_ell


class SearchExperiment(object):
    """
    Class that handles searching FRBs in one experiment.
    """

    def __init__(self, exp_code, cfx_file, dsp_params, raw_data_dir, db_file):
        """
        :param exp_code:
            Experiment code.
        :param cfx_file:
            Path to experiment CFX file.
        :param dsp_params:
            Parameters of dynamical spectra to create. Dictionary with the
            following keys: ``n_nu`` [ex. 64], ``nu_0`` [MHz], ``d_t``  [s],
            ``d_nu`` [MHz].
        :param raw_data_dir:
            Directory with subdirectories that contains raw data for all
            antennas.
        :param db_file:
            Path to DB file.
        """
        self.exp_code = exp_code
        self.cfx_file = cfx_file
        self.dsp_params = dsp_params
        self.raw_data_dir = raw_data_dir
        self.db_file = db_file
        self.cfx = CFX(cfx_file)

    @property
    def exp_params(self):
        """
        Returns dictionary with key - raw data file name & value - parameters.
        """
        return self.cfx.parse_cfx(self.exp_code)

    def dsp_generator(self, m5_file, m5_params, chunk_size=100.):
        """
        Generator that returns instances of ``DynSpectra`` class.

        :param m5_file:
            Raw data file in M5 format.
        :param m5_params:
            Dictionary with meta data.
        :param chunk_size: (optional)
            Size (in s) of chunks to process raw data. (default: ``100.``)
        """
        dsp_params = self.dsp_params
        dsp_params.update({'offset': 0., 'outfile': None, 'dur': chunk_size})
        m5file_fmt = m5_params[2]
        cfx_fmt = m5_params[-1]
        m5 = M5(m5_file, m5file_fmt)
        offset = 0

        while offset * 32e6 < m5.size:
            dsp_params.update({'offst': offset})
            ds = m5.create_dspec(**dsp_params)

            # NOTE: all 4 channels are stacked forming dsarr:
            dsarr = dspec_cat(os.path.basename(ds['Dspec_file']),
                              cfx_fmt)
            metadata = ds
            metadata['Raw_data_file'] = m5_file
            metadata['Exp_data'] = m5_params
            t_0 = m5.start_time + TimeDelta(offset, format='sec')

            dsp = DynSpectra(dsp_params['n_nu'],
                             int(chunk_size/dsp_params['d_t']),
                             dsp_params['nu_0'], dsp_params['d_nu'],
                             dsp_params['d_t'], meta_data=metadata, t_0=t_0)
            dsp.add_values(dsarr)
            offset += offset

            yield dsp

    # TODO: Add checking DB if searching for FRBs with the same set of
    # de-dispersion + pre-processing + searching parameters was already done
    # before.
    def run(self, de_disp_params, pre_process_params, search_params,
            antenna=None, except_antennas=None, cache_dir=None):
        """
        Run pipeline on experiment.

        :param de_disp_params:
            Dictionary with de-dispersion parameters.
        :param pre_process_params:
            Dictionary with pre-processing parameters.
        :param search_params:
            Dictionary with searching parameters.
        :param antenna: (optional)
            Antenna to search. If ``None`` then search all available. (default:
            ``None``)
        :param except_antennas: (optional)
            Antennas not to search. If ``None`` then search all available.
            (default: ``None``)
        :param cache_dir: (optional)
            Directory to store cache HDF5 files. If ``None`` - use CWD.
            (default: ``None``)

        :note:
            Argument dictionaries should have keys: 'func', 'args', 'kwargs'
            with corresponding values passed to ``Searcher.run`` method.
        """
        # Dict with keys - antennas & values - list of ``Candidate`` instances
        # for given antenna detected.
        exp_candidates = defaultdict(list)
        for m5_file, m5_params in self.exp_params.items():
            m5_antenna = m5_params['antenna']
            if antenna and antenna != m5_antenna:
                continue
            if except_antennas and m5_antenna in except_antennas:
                continue
            dsp_gen = self.dsp_generator(m5_file, m5_params)
            for dsp in dsp_gen:
                searcher = Searcher(dsp, cache_dir=cache_dir)
                candidates = searcher.run(de_disp_params['func'],
                                          search_func=search_params['func'],
                                          preprocess_func=pre_process_params['func'],
                                          de_disp_args=de_disp_params['args'],
                                          de_disp_kwargs=de_disp_params['kwargs'],
                                          search_args=search_params['args'],
                                          search_kwargs=search_params['kwargs'],
                                          preprocess_args=pre_process_params['args'],
                                          preprocess_kwargs=pre_process_params['kwargs'],
                                          db_file=self.db_file)
                if candidates:
                    exp_candidates[dsp.meta_data['antenna']].extend(candidates)
        return exp_candidates


if __name__ == '__main__':
    exp_code = 'raks12ec'
    cfx_file = '/home/ilya/code/frb/frb/RADIOASTRON_RAKS12EC_C_20151030T210000_ASC_V1.cfx'
    raw_data_dir = '/mnt/frb_data/raw_data/2015_303_raks12ec'
    db_file = '/home/ilya/code/frb/frb/frb.db'
    # Step used in de-dispersion
    d_dm = 30.
    # Values of DM to de-disperse
    dm_grid = np.arange(0., 1000., d_dm)

    # Arguments for searching function
    search_kwargs = {'x_stddev': 6.,
                     'y_to_x_stddev': 0.3,
                     'theta_lims': [130., 180.],
                     'x_cos_theta': 3.,
                     'd_dm': d_dm,
                     'amplitude': 3},
    # Arguments for pre-processing function
    preprocess_kwargs = {'disk_size': 3,
                         'threshold_big_perc': 90.,
                         'threshold_perc': 97.5,
                         'statistic': 'mean'}
    de_disp_params = {'func': de_disperse_cumsum,
                      'args': [dm_grid]}
    pre_process_params = {'func': create_ellipses,
                          'kwargs': preprocess_kwargs}
    search_params = {'func': search_candidates_ell,
                     'kwargs': search_kwargs}

    # Create pipeline
    dsp_params = {'n_nu': 64, 'nu_0': 4836., 'd_t': 0.001}
    pipeline = SearchExperiment(exp_code, cfx_file, dsp_params, raw_data_dir,
                                db_file)
    # Run pipeline with given parameters
    candidates_dict = pipeline.run(de_disp_params, pre_process_params,
                                   search_params)

    session = connect_to_db(db_file)
    # Query DB
    frb_list = query_frb(session, exp_code, d_dm=200., d_t=0.1)
    print "Found FRBs:"
    for frb in frb_list:
        print frb




