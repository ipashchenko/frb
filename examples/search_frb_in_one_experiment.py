import numpy as np
from astropy.time import Time, TimeDelta
from frb.dyn_spectra import create_from_txt
from frb.search_candidates import Searcher
from frb.dedispersion import de_disperse_cumsum
from frb.search import search_candidates_ell, create_ellipses
from frb.queries import query_frb, connect_to_db


# Set random generator seed for reproducibility
np.random.seed(1)
# DB file
db_file = '/home/ilya/code/akutkin/frb/frb/frb.db'

# Real data from WB
txt = '/home/ilya/code/akutkin/frb/data/100_sec_wb_raes08a_128ch.asc'
meta_data = {'antenna': None, 'freq': 'l', 'band': 'u', 'pol': 'r',
             'exp_code': 'raks00'}
t0 = Time.now()
mother_dsp = create_from_txt(txt, 1684., 16. / 128, 0.001, meta_data, t0)

antennas = ['AR', 'EF', 'RA']
slices = [(0., 0.3), (0.3, 0.7), (0.7, 1)]
# Step of de-dispersion
d_dm = 30.

n_real = 2
# Time of `real` FRB
t_0_reals = np.linspace(0, 30, n_real+2)[1:-1]
amp_reals = np.random.uniform(0.25, 0.35, size=n_real)
width_reals = np.random.uniform(0.001, 0.003, size=n_real)
dm_value_reals = np.random.uniform(100, 500, size=n_real)

for t_0_real, amp_real, width_real, dm_value_real in zip(t_0_reals, amp_reals,
                                                         width_reals,
                                                         dm_value_reals):
    t_1 = t0 + TimeDelta(t_0_real, format='sec')
    print "REAL FRBs are" \
          " t0={:%Y-%m-%d %H:%M:%S.%f},".format(t_1.utc.datetime)[:-3] + \
          " amp={:.2f}, width={:.4f}, dm={:.0f}".format(amp_real, width_real,
                                                        dm_value_real)
# Values of DM to de-disperse
dm_grid = np.arange(0., 1000., d_dm)

for antenna, ant_slice in zip(antennas, slices):
    print "Loading dynamical dpectra for antenna {}".format(antenna)
    dsp = mother_dsp.slice(*ant_slice)
    dsp.meta_data.update({'antenna': antenna})
    print "Adding REAL FRBs to {} data".format(antenna)
    for t_0_real, amp_real, width_real, dm_value_real in zip(t_0_reals,
                                                             amp_reals,
                                                             width_reals,
                                                             dm_value_reals):
        dsp.add_pulse(t_0_real, amp_real, width_real, dm_value_real)

    # Initialize searcher class
    searcher = Searcher(dsp)
    # Run search for FRB with some parameters of de-dispersion, pre-processing,
    # searching algorithms
    candidates = searcher.run(de_disp_func=de_disperse_cumsum,
                              search_func=search_candidates_ell,
                              preprocess_func=create_ellipses,
                              de_disp_args=[dm_grid],
                              search_kwargs={'x_stddev': 6.,
                                             'y_to_x_stddev': 0.3,
                                             'theta_lims': [130., 180.],
                                             'x_cos_theta': 3.,
                                             'd_dm': d_dm,
                                             'amplitude': 3},
                              preprocess_kwargs={'disk_size': 3,
                                                 'threshold_big_perc': 90.,
                                                 'threshold_perc': 97.5,
                                                 'statistic': 'mean'},
                              db_file=db_file)
    print "Found {} candidates".format(len(candidates))
    for candidate in candidates:
        print candidate


session = connect_to_db(db_file)
# Query DB
frb_list = query_frb(session, meta_data['exp_code'], d_dm=200., d_t=0.1)
print "Found FRBs:"
for frb in frb_list:
    print frb

