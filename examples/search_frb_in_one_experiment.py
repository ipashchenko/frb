import numpy as np
from astropy.time import Time
from frb.frames import create_from_txt
from frb.search_candidates import Searcher
from frb.dedispersion import de_disperse_cumsum
from frb.search import search_candidates_ell, create_ellipses
from frb.queries import query_frb, connect_to_db


# Set random generator seed for reproducibility
np.random.seed(123)

# Real data from WB
txt = '/home/ilya/code/akutkin/frb/data/100_sec_wb_raes08a_128ch.asc'
mother_frame = create_from_txt(txt, 1684., 0, 16./128, 0.001)
exp_code = 'raks00'
antennas = ['AR', 'EF', 'RA']
slices = [(0., 0.3), (0.3, 0.7), (0.7, 1)]
# Step of de-dispersion
d_dm = 30.

# Zero time
t = Time.now()
print "Zero time: {}".format(t)
n_real = 5
# Time of `real` FRB
t_0_reals = np.random.uniform(1, 20, size=n_real)
amp_reals = np.random.uniform(0.45, 0.50, size=n_real)
width_reals = np.random.uniform(0.001, 0.005, size=n_real)
dm_value_reals = np.random.uniform(0, 1000, size=n_real)

for t_0_real, amp_real, width_real, dm_value_real in zip(t_0_reals, amp_reals,
                                                         width_reals,
                                                         dm_value_reals):
    print "REAL FRB: t={}, A={}, W={}, DM={}".format(t_0_real, amp_real,
                                                     width_real, dm_value_real)
# Values of DM to de-disperse
dm_grid = np.arange(0., 1000., d_dm)

for antenna, ant_slice in zip(antennas, slices):
    print "Creating Dynamical Spectra for antenna {}".format(antenna)
    frame = mother_frame.slice(*ant_slice)
    print "Adding REAL FRBs to {} data".format(antenna)
    for t_0_real, amp_real, width_real, dm_value_real in zip(t_0_reals,
                                                             amp_reals,
                                                             width_reals,
                                                             dm_value_reals):
        frame.add_pulse(t_0_real, amp_real, width_real, dm_value_real)

    meta_data = {'antenna': antenna, 'freq': 'L', 'band': 'U', 'pol': 'R',
                 'exp_code': 'raks00', 'nu_max': 1684., 't_0': t,
                 'd_nu': 16./128., 'd_t': 0.001}
    # Initialize searcher class
    searcher = Searcher(dsp=frame.values, meta_data=meta_data)
    # Run search for FRB with some parameters of de-dispersion, pre-processing,
    # searching algorithms
    candidates = searcher.run(de_disp_func=de_disperse_cumsum,
                              search_func=search_candidates_ell,
                              preprocess_func=create_ellipses,
                              de_disp_args=[dm_grid],
                              search_kwargs={'x_stddev': 10.,
                                             'y_to_x_stddev': 0.3,
                                             'theta_lims': [130., 180.],
                                             'x_cos_theta': 5.,
                                             'd_dm': d_dm,
                                             'amplitude': 10},
                              preprocess_kwargs={'disk_size': 3,
                                                 'threshold_big_perc': 90.,
                                                 'threshold_perc': 97.5,
                                                 'statistic': 'mean'},
                              db_file='/home/ilya/code/akutkin/frb/frb/frb.db')
    print "Found {} candidates".format(len(candidates))
    for candidate in candidates:
        print candidate


session = connect_to_db("/home/ilya/code/akutkin/frb/frb/frb.db")
# Query DB
frb_list = query_frb(session, exp_code, d_dm=200., d_t=0.1)
print "Found FRBs:"
for frb in frb_list:
    print frb

