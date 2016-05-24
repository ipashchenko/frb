import numpy as np
from cfx import CFX
from raw_data import M5
from queries import connect_to_db, query_frb
from search_candidates import Searcher
from dedispersion import de_disperse_cumsum
from search import create_ellipses, search_candidates_ell


class SearchExperiment(object):
    """
    Class that handles searching FRBs in one experiment.
    """

    def __init__(self, exp_code, cfx_file, raw_data_dir, db_file):
        """
        :param exp_code:
            Experiment code.
        :param cfx_file:
            Path to experiment CFX file.
        :param raw_data_dir:
            Directory with subdirectories that contains raw data for all
            antennas.
        :param db_file:
            Path to DB file.
        """
        self.exp_code = exp_code
        self.cfx_file = cfx_file
        self.raw_data_dir = raw_data_dir
        self.db_file = db_file
        self.cfx = CFX(cfx_file)

    @property
    def exp_params(self):
        """
        Returns dictionary with key - raw data file name & value - parameters.
        """
        return self.cfx.parse_cfx(self.exp_code)

    def dsp_generator(self, m5_file, m5_params):
        """
        Generator that returns dsp arrays with dynamical spectra and metadata
        dictionary for each dynamical spectra.

        :param m5_file:
            Raw data file in M5 format.
        :param m5_params:
            Dictionary with meta data.
        """
        raise NotImplementedError

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
        :param cache_dir: (optional)
            Directory to store cache HDF5 files. If ``None`` - use CWD.
            (default: ``None``)

        :note:
            Argument dictionaries should have keys: 'func', 'args', 'kwargs'
            with corresponding values passed to ``Searcher.run`` method.
        """
        for m5_file, m5_params in self.exp_params.items():
            m5_antenna = m5_params['antenna']
            if antenna and antenna != m5_antenna:
                continue
            if except_antennas and m5_antenna in except_antennas:
                continue
            dsp_gen = self.dsp_generator(m5_file, m5_params)
            for dsp, dsp_param in dsp_gen:
                searcher = Searcher(dsp, dsp_param, cache_dir=cache_dir)
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

        session = connect_to_db(self.db_file)
        # Query DB
        frb_list = query_frb(session, self.exp_code, d_dm=200., d_t=0.1)
        print "Found FRBs:"
        for frb in frb_list:
            print frb


if __name__ == '__main__':
    exp_code = 'raks12er'
    cfx_file = None
    raw_data_dir = None
    db_file = '/home/ilya/code/akutkin/frb/frb/frb.db'
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
    pipeline = SearchExperiment(exp_code, cfx_file, raw_data_dir, db_file)
    # Run pipeline with given parameters
    pipeline.run(de_disp_params, pre_process_params, search_params)




